# ================================================================
#                      analysis_chains.py (ICT ПРАВИЛЬНАЯ ВЕРСИЯ)
#      
#      СТОП ТОЛЬКО ЗА:
#      1. SWING HIGH (для SHORT)
#      2. SWING LOW (для LONG)  
#      3. ORDER BLOCK (если swing не найден)
#
#      НИКОГДА НЕ СТАВИМ СТОП "В ВОЗДУХЕ"!
# ================================================================

from __future__ import annotations
from typing import List, Optional
from enum import Enum

from analysis_interfaces import ChainContext, ChainSignal, Zone, DetectionResult


# ---------------------------------------------------------------
# Direction enum
# ---------------------------------------------------------------
class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


# ---------------------------------------------------------------
#    Zone type helpers
# ---------------------------------------------------------------

def _is_fvg(z: Zone) -> bool:
    return "FVG" in z.type.upper()


def _is_ob(z: Zone) -> bool:
    return "OB" in z.type.upper()


def _is_fh(z: Zone) -> bool:
    return "FH_" in z.type.upper()


def _is_idm(z: Zone) -> bool:
    return "IDM" in z.type.upper()


def _is_liquidity(z: Zone) -> bool:
    t = z.type.upper()
    return any(x in t for x in ["EQH", "EQL", "SWEEP", "BSL", "SSL", "LIQUIDITY"])


def _zone_direction(z: Zone) -> Direction:
    t = z.type.upper()
    if "UP" in t or "BULL" in t:
        return Direction.LONG
    if "DOWN" in t or "BEAR" in t:
        return Direction.SHORT
    return Direction.LONG


# ---------------------------------------------------------------
#     Zone validation helpers
# ---------------------------------------------------------------

def _is_zone_inside(inner: Zone, outer: Zone, tolerance: float = 0.0) -> bool:
    outer_range = outer.high - outer.low
    allowed_overflow = outer_range * tolerance
    return (inner.low >= outer.low - allowed_overflow and
            inner.high <= outer.high + allowed_overflow)


def _zones_overlap(z1: Zone, z2: Zone) -> bool:
    return not (z1.high < z2.low or z1.low > z2.high)


def _get_primary_zone(detection: DetectionResult, zone_types: List[str] = None) -> Optional[Zone]:
    if not detection or not detection.zones:
        return None

    if not zone_types:
        zone_types = ["OB", "FVG", "IDM"]

    for zone in detection.zones:
        for zt in zone_types:
            if zt in zone.type.upper():
                return zone

    return None


# ---------------------------------------------------------------
#     ICT SWING FINDER
#     
#     По ICT методологии:
#     SWING HIGH = свеча где HIGH ВЫШЕ чем у left свечей слева И right свечей справа
#     SWING LOW = свеча где LOW НИЖЕ чем у left свечей слева И right свечей справа
# ---------------------------------------------------------------

def _find_ict_swings(candles: List, left: int = 2, right: int = 1) -> List[dict]:
    """
    ICT Swing High/Low finder.
    
    Параметры:
    - left=2, right=1: стандартная 3-свечная формация
    - left=3, right=2: более значимые свинги
    
    Returns:
        List of {"kind": "HIGH"/"LOW", "index": int, "price": float}
    """
    swings = []
    n = len(candles)

    if n < left + right + 1:
        return swings

    for i in range(left, n - right):
        c = candles[i]

        # ICT Swing High
        is_swing_high = True
        for j in range(i - left, i):
            if candles[j].high >= c.high:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(i + 1, i + right + 1):
                if candles[j].high >= c.high:
                    is_swing_high = False
                    break

        # ICT Swing Low
        is_swing_low = True
        for j in range(i - left, i):
            if candles[j].low <= c.low:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(i + 1, i + right + 1):
                if candles[j].low <= c.low:
                    is_swing_low = False
                    break

        if is_swing_high:
            swings.append({
                "kind": "HIGH",
                "index": i,
                "price": c.high,
            })
        if is_swing_low:
            swings.append({
                "kind": "LOW",
                "index": i,
                "price": c.low,
            })

    return swings


def _calculate_atr(candles: List, period: int = 14) -> float:
    """ATR для буфера"""
    if len(candles) < 2:
        return 0.0

    trs = []
    for i in range(1, min(period + 1, len(candles))):
        h = candles[-i].high
        l = candles[-i].low
        prev_close = candles[-i - 1].close
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)

    return sum(trs) / len(trs) if trs else 0.0


# ---------------------------------------------------------------
#     ICT STOP LOSS - ПРИОРИТЕТ: OB 1h/4h → SWING (мин 1.5%)
# ---------------------------------------------------------------

def _find_ict_stop_loss(
        ctx: ChainContext,
        tf: str,
        direction: Direction,
        entry: float
) -> Optional[float]:
    """
    ICT Stop Loss - ТОЛЬКО за структурой!
    
    ПРИОРИТЕТ:
    1. Order Block на 1h или 4h - САМЫЙ НАДЁЖНЫЙ
    2. Swing High/Low с минимальным риском 1.5%
    3. None → ОТКЛОНЯЕМ сигнал (не ставим в воздухе!)
    
    МИНИМАЛЬНЫЙ РИСК: 1.5%
    МАКСИМАЛЬНЫЙ РИСК: 8%
    """

    DEBUG = True
    MIN_RISK_PCT = 1.5  # Минимальный риск 1.5%
    MAX_RISK_PCT = 8.0  # Максимальный риск 8%

    def dbg(msg):
        if DEBUG:
            print(f"      [ICT-SL] {msg}")

    dbg(f"Finding SL for {direction} @ entry {entry:.5f}")
    dbg(f"Min risk: {MIN_RISK_PCT}%, Max risk: {MAX_RISK_PCT}%")

    min_sl_distance = entry * MIN_RISK_PCT / 100
    max_sl_distance = entry * MAX_RISK_PCT / 100

    # Буфер (небольшой отступ от уровня)
    candles_current = ctx.candles.get(tf, [])
    atr = _calculate_atr(candles_current, 14) if candles_current else entry * 0.01
    buffer = max(atr * 0.15, entry * 0.001)
    
    dbg(f"Buffer: {buffer:.5f}, Min SL dist: {min_sl_distance:.5f}")

    stop_loss = None
    sl_source = None

    # =================================================================
    # ПРИОРИТЕТ 1: ORDER BLOCK на 1h или 4h
    # =================================================================
    dbg("Checking Order Blocks (1h, 4h)...")
    
    all_obs = []
    for check_tf in ["4h", "1h"]:  # ТОЛЬКО 1h и 4h!
        det = ctx.detections.get(check_tf)
        if det and det.zones:
            for z in det.zones:
                if _is_ob(z):
                    all_obs.append((z, check_tf))
    
    dbg(f"Found {len(all_obs)} OBs on 1h/4h")

    if all_obs:
        if direction == Direction.LONG:
            # LONG: ищем Bullish OB НИЖЕ entry с достаточным риском
            valid_obs = [(z, htf) for z, htf in all_obs 
                         if "BULL" in z.type.upper() 
                         and z.low < entry
                         and (entry - z.low) >= min_sl_distance
                         and (entry - z.low) <= max_sl_distance]
            
            if valid_obs:
                # Берём ближайший с достаточным риском
                valid_obs.sort(key=lambda x: entry - x[0].low)
                chosen_ob, chosen_tf = valid_obs[0]
                stop_loss = chosen_ob.low - buffer
                sl_source = f"OB {chosen_tf}"
                risk_pct = (entry - stop_loss) / entry * 100
                dbg(f"✓ Using {chosen_tf} Bullish OB: {chosen_ob.low:.5f}-{chosen_ob.high:.5f}, Risk: {risk_pct:.2f}%")
                
        else:  # SHORT
            valid_obs = [(z, htf) for z, htf in all_obs 
                         if "BEAR" in z.type.upper() 
                         and z.high > entry
                         and (z.high - entry) >= min_sl_distance
                         and (z.high - entry) <= max_sl_distance]
            
            if valid_obs:
                valid_obs.sort(key=lambda x: x[0].high - entry)
                chosen_ob, chosen_tf = valid_obs[0]
                stop_loss = chosen_ob.high + buffer
                sl_source = f"OB {chosen_tf}"
                risk_pct = (stop_loss - entry) / entry * 100
                dbg(f"✓ Using {chosen_tf} Bearish OB: {chosen_ob.low:.5f}-{chosen_ob.high:.5f}, Risk: {risk_pct:.2f}%")

    # =================================================================
    # ПРИОРИТЕТ 2: SWING HIGH/LOW (если OB не найден)
    # =================================================================
    if stop_loss is None:
        dbg("No valid OB, checking Swings...")
        
        all_swings = []
        for check_tf in ["4h", "1h", "15m"]:
            candles = ctx.candles.get(check_tf, [])
            if not candles or len(candles) < 15:
                continue
            
            # Строгие параметры для значимых свингов
            if check_tf == "4h":
                swings = _find_ict_swings(candles, left=3, right=2)
            elif check_tf == "1h":
                swings = _find_ict_swings(candles, left=3, right=2)
            else:  # 15m
                swings = _find_ict_swings(candles, left=5, right=3)
            
            for s in swings:
                s["tf"] = check_tf
                all_swings.append(s)
            
            dbg(f"{check_tf}: {len(swings)} swings")

        if direction == Direction.LONG:
            # LONG: Swing Low НИЖЕ entry с минимальным риском 1.5%
            valid_lows = [s for s in all_swings 
                          if s["kind"] == "LOW" 
                          and s["price"] < entry
                          and (entry - s["price"]) >= min_sl_distance
                          and (entry - s["price"]) <= max_sl_distance]
            
            if valid_lows:
                valid_lows.sort(key=lambda s: entry - s["price"])
                
                dbg(f"Valid swing lows (risk >= {MIN_RISK_PCT}%):")
                for i, s in enumerate(valid_lows[:3]):
                    dist_pct = (entry - s["price"]) / entry * 100
                    dbg(f"  {i+1}. {s['tf']} @ {s['price']:.5f} ({dist_pct:.2f}%)")
                
                chosen = valid_lows[0]
                stop_loss = chosen["price"] - buffer
                sl_source = f"Swing Low {chosen['tf']}"
            else:
                dbg(f"No swing lows with risk >= {MIN_RISK_PCT}%")
                
        else:  # SHORT
            valid_highs = [s for s in all_swings 
                           if s["kind"] == "HIGH" 
                           and s["price"] > entry
                           and (s["price"] - entry) >= min_sl_distance
                           and (s["price"] - entry) <= max_sl_distance]
            
            if valid_highs:
                valid_highs.sort(key=lambda s: s["price"] - entry)
                
                dbg(f"Valid swing highs (risk >= {MIN_RISK_PCT}%):")
                for i, s in enumerate(valid_highs[:3]):
                    dist_pct = (s["price"] - entry) / entry * 100
                    dbg(f"  {i+1}. {s['tf']} @ {s['price']:.5f} ({dist_pct:.2f}%)")
                
                chosen = valid_highs[0]
                stop_loss = chosen["price"] + buffer
                sl_source = f"Swing High {chosen['tf']}"
            else:
                dbg(f"No swing highs with risk >= {MIN_RISK_PCT}%")

    # =================================================================
    # ФИНАЛЬНЫЕ ПРОВЕРКИ
    # =================================================================
    if stop_loss is None:
        dbg("❌ NO VALID STRUCTURE FOR SL - REJECTING SIGNAL!")
        return None

    # Проверка направления
    if direction == Direction.LONG and stop_loss >= entry:
        dbg(f"❌ Invalid: SL {stop_loss:.5f} >= Entry {entry:.5f}")
        return None
    if direction == Direction.SHORT and stop_loss <= entry:
        dbg(f"❌ Invalid: SL {stop_loss:.5f} <= Entry {entry:.5f}")
        return None

    risk = abs(entry - stop_loss)
    risk_pct = risk / entry * 100
    
    # Финальная проверка на минимальный риск
    if risk_pct < MIN_RISK_PCT:
        dbg(f"❌ Risk {risk_pct:.2f}% < {MIN_RISK_PCT}% - REJECTING!")
        return None

    dbg(f"✓ FINAL SL: {stop_loss:.5f} ({sl_source}), Risk: {risk_pct:.2f}%")
    return stop_loss


# ---------------------------------------------------------------
#     TP FINDER
# ---------------------------------------------------------------

def _find_liquidity_targets(
        ctx: ChainContext,
        tf: str,
        direction: Direction,
        entry: float,
        risk: float
) -> List[float]:
    """Находит TP на liquidity/swing points"""
    targets = []

    # Собираем зоны
    all_zones = []
    for check_tf in ["15m", "1h", "4h", "1d"]:
        det = ctx.detections.get(check_tf)
        if det and det.zones:
            for z in det.zones:
                all_zones.append((z, check_tf))

    # Собираем свинги
    all_swings = []
    for check_tf in ["15m", "1h", "4h"]:
        candles = ctx.candles.get(check_tf, [])
        if candles and len(candles) > 10:
            swings = _find_ict_swings(candles, left=2, right=1)
            all_swings.extend(swings)

    if direction == Direction.LONG:
        # Liquidity выше
        liq = [(z, htf) for z, htf in all_zones if _is_liquidity(z) and z.low > entry]
        for z, htf in liq:
            targets.append(z.low)

        # Swing highs
        highs = [s["price"] for s in all_swings if s["kind"] == "HIGH" and s["price"] > entry]
        targets.extend(highs)

        targets = sorted(set(targets))

    else:  # SHORT
        liq = [(z, htf) for z, htf in all_zones if _is_liquidity(z) and z.high < entry]
        for z, htf in liq:
            targets.append(z.high)

        lows = [s["price"] for s in all_swings if s["kind"] == "LOW" and s["price"] < entry]
        targets.extend(lows)

        targets = sorted(set(targets), reverse=True)

    # Фильтр: минимум 1.5R
    min_dist = risk * 1.5
    if direction == Direction.LONG:
        valid = [t for t in targets if t - entry >= min_dist]
    else:
        valid = [t for t in targets if entry - t >= min_dist]

    return valid[:10]


# ---------------------------------------------------------------
#     Price in zone checker
# ---------------------------------------------------------------

def _is_price_in_zone(ctx: ChainContext, zone: Zone, tf: str) -> bool:
    candles = ctx.candles.get(tf, [])
    if not candles:
        return False

    current_price = candles[-1].close
    zone_size = zone.high - zone.low
    buffer = min(zone_size * 0.5, current_price * 0.005)

    return (zone.low - buffer) <= current_price <= (zone.high + buffer)


# ---------------------------------------------------------------
#     SIGNAL BUILDER
# ---------------------------------------------------------------

def _make_signal(
        ctx: ChainContext,
        chain_id: str,
        direction: Direction,
        zone: Zone,
        tf: Optional[str] = None,
        description: str = "",
) -> Optional[ChainSignal]:
    """Строит сигнал с ICT stop loss"""
    tf = tf or zone.tf

    DEBUG = True

    def dbg(msg):
        if DEBUG:
            print(f"    [{chain_id}] {msg}")

    # BIAS
    if ctx.bias_contexts and tf in ctx.bias_contexts:
        bias = ctx.bias_contexts[tf]
        if bias.bias == "STRONG_BEARISH" and direction == Direction.LONG:
            dbg(f"REJECTED: STRONG_BEARISH bias")
            return None
        if bias.bias == "STRONG_BULLISH" and direction == Direction.SHORT:
            dbg(f"REJECTED: STRONG_BULLISH bias")
            return None

    if not _is_price_in_zone(ctx, zone, tf):
        dbg(f"REJECTED: price not in zone")
        return None

    low, high = zone.low, zone.high
    dbg(f"Zone: {zone.type} @ {low:.5f}-{high:.5f}")

    # ENTRY
    if "OB" in zone.type:
        entry = low + (high - low) * 0.62 if direction == Direction.LONG else high - (high - low) * 0.62
    elif "FVG" in zone.type:
        entry = (low + high) / 2.0
    else:
        entry = (low + high) / 2.0

    dbg(f"Entry: {entry:.5f}, Direction: {direction}")

    # STOP LOSS - ТОЛЬКО ЗА SWING/OB!
    sl = _find_ict_stop_loss(ctx, tf, direction, entry)

    if sl is None:
        dbg(f"REJECTED: no valid structure for SL")
        return None

    risk = abs(entry - sl)
    risk_pct = risk / entry * 100
    dbg(f"SL: {sl:.5f}, Risk: {risk_pct:.2f}%")

    # TAKE PROFITS
    targets = _find_liquidity_targets(ctx, tf, direction, entry, risk)

    if targets:
        tp1 = targets[0]
        tp2 = targets[1] if len(targets) > 1 else (tp1 + risk if direction == Direction.LONG else tp1 - risk)
    else:
        tp1 = entry + risk * 2 if direction == Direction.LONG else entry - risk * 2
        tp2 = entry + risk * 3.5 if direction == Direction.LONG else entry - risk * 3.5

    # RR
    rr = abs(tp1 - entry) / risk if risk > 0 else 0.0
    dbg(f"TP1: {tp1:.5f}, RR: {rr:.2f}")

    if rr < 1.5:
        dbg(f"REJECTED: RR {rr:.2f} < 1.5")
        return None

    # Validation
    if direction == Direction.LONG:
        if tp1 <= entry or sl >= entry:
            return None
    else:
        if tp1 >= entry or sl <= entry:
            return None

    dbg(f"✓ SIGNAL OK!")

    return ChainSignal(
        symbol=ctx.symbol,
        chain_id=chain_id,
        tf=tf,
        direction=direction,
        entry=entry,
        stop_loss=sl,
        take_profits=[tp1, tp2],
        rr=rr,
        description=f"{description} | SL: Behind Swing/OB",
    )


# ================================================================
#               ЦЕПОЧКИ
# ================================================================

class Chain_1_1:
    chain_id = "1.1"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_d = _get_primary_zone(ctx.detections.get("1d"), ["OB", "FVG"])
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if (_is_ob(z) or _is_fvg(z)) and _is_zone_inside(z, zone_d):
                zone_4h = z
                break

        if not zone_4h:
            return []

        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break

        if not zone_15m:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_15m, "15m", "D->4h->1h->15m")
        return [sig] if sig else []


class Chain_1_2:
    chain_id = "1.2"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_d_idm = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_idm(z):
                zone_d_idm = z
                break
        if not zone_d_idm:
            return []

        direction = _zone_direction(zone_d_idm)

        zone_4h_idm = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_idm(z):
                zone_4h_idm = z
                break
        if not zone_4h_idm:
            return []

        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break
        if not zone_15m:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_15m, "15m", "D(IDM)->4h(IDM)->1h->15m")
        return [sig] if sig else []


class Chain_1_3:
    chain_id = "1.3"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_d = _get_primary_zone(ctx.detections.get("1d"), ["OB", "FVG"])
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        idm_4h = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_idm(z):
                idm_4h = z
                break
        if not idm_4h:
            return []

        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break
        if not zone_15m:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_15m, "15m", "D->4h(IDM)->1h->15m")
        return [sig] if sig else []


class Chain_1_4:
    chain_id = "1.4"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_d = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_d = z
                break
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if (_is_ob(z) or _is_fvg(z)) and _is_zone_inside(z, zone_d):
                zone_4h = z
                break
        if not zone_4h:
            return []

        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break
        if not zone_15m:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_15m, "15m", "D(FVG)->4h->1h->15m")
        return [sig] if sig else []


class Chain_1_5:
    chain_id = "1.5"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_d = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_d = z
                break
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        candles_d = ctx.candles.get("1d", [])
        if not candles_d:
            return []

        last_d = candles_d[-1]
        in_fvg = False
        if direction == Direction.LONG:
            if last_d.low <= zone_d.high and last_d.close >= zone_d.low:
                in_fvg = True
        else:
            if last_d.high >= zone_d.low and last_d.close <= zone_d.high:
                in_fvg = True
        if not in_fvg:
            return []

        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if _is_fvg(z) and _is_zone_inside(z, zone_d, tolerance=0.2):
                zone_4h = z
                break
        if not zone_4h:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_4h, "4h", "D(FVG)->reaction->4h")
        return [sig] if sig else []


class Chain_2_6:
    chain_id = "2.6"

    def _get_poi(self, candles, idx) -> Optional[Zone]:
        if idx >= len(candles):
            return None
        c = candles[idx]
        if c.close > c.open:
            poi_low, poi_high = max(c.open, c.close), c.high
        else:
            poi_low, poi_high = c.low, min(c.open, c.close)
        if poi_high <= poi_low:
            return None
        return Zone(tf="4h", low=poi_low, high=poi_high, type="POI")

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        fh_zones = [z for z in ctx.detections.get("4h", DetectionResult([], None)).zones if _is_fh(z)]
        if not fh_zones:
            return []

        fh_zone = fh_zones[-1]
        direction = Direction.SHORT if "HIGH" in fh_zone.type.upper() else Direction.LONG

        candles_4h = ctx.candles.get("4h", [])
        if not candles_4h:
            return []

        fractal_idx = None
        for i in range(len(candles_4h) - 1, max(0, len(candles_4h) - 20), -1):
            c = candles_4h[i]
            if "HIGH" in fh_zone.type:
                if fh_zone.low <= c.high <= fh_zone.high:
                    fractal_idx = i
                    break
            else:
                if fh_zone.low <= c.low <= fh_zone.high:
                    fractal_idx = i
                    break
        if fractal_idx is None:
            return []

        poi_zone = self._get_poi(candles_4h, fractal_idx)
        if not poi_zone:
            return []

        poi_touched = any(c.low <= poi_zone.high and c.high >= poi_zone.low for c in candles_4h[-5:])
        if not poi_touched:
            return []

        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _zone_direction(z) == direction and _zones_overlap(z, poi_zone):
                zone_15m = z
                break
        if not zone_15m:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_15m, "15m", "Sweep->POI->15m")
        return [sig] if sig else []


class Chain_3_2:
    chain_id = "3.2"

    def __init__(self):
        self.visited = {}

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_4h = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_4h = z
                break
        if not zone_4h:
            return []

        key = f"{zone_4h.low:.2f}_{zone_4h.high:.2f}"
        if key in self.visited:
            return []

        direction = _zone_direction(zone_4h)
        candles_4h = ctx.candles.get("4h", [])
        if not candles_4h:
            return []

        last = candles_4h[-1]
        entered = False
        if direction == Direction.LONG:
            if last.low <= zone_4h.high:
                entered = True
                if last.close > zone_4h.high:
                    return []
        else:
            if last.high >= zone_4h.low:
                entered = True
                if last.close < zone_4h.low:
                    return []
        if not entered:
            return []

        self.visited[key] = True

        zone_1h = None
        for z in ctx.detections.get("1h", DetectionResult([], None)).zones:
            if _is_fvg(z) and _zone_direction(z) == direction:
                zone_1h = z
                break
        if not zone_1h:
            return []

        sig = _make_signal(ctx, self.chain_id, direction, zone_1h, "1h", "First 4h FVG->1h")
        return [sig] if sig else []


class Signal_1:
    chain_id = "Signal_1"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        zone_fh = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fh(z):
                zone_fh = z
                break
        if not zone_fh:
            return []

        zone_ob = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_ob(z):
                zone_ob = z
                break
        if not zone_ob:
            return []

        direction = _zone_direction(zone_ob)
        sig = _make_signal(ctx, self.chain_id, direction, zone_ob, "4h", "FH.D + OB.4h")
        return [sig] if sig else []