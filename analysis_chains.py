# ================================================================
#                      analysis_chains.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
#      SL - ТОЛЬКО за swing high/low или POI (1h/4h)
#      TP - ТОЛЬКО на liquidity или imbalance tests
#      БЕЗ процентных лимитов на SL!
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
    """Проверяет, является ли зона ликвидностью (EQH/EQL/sweep/BSL/SSL)"""
    t = z.type.upper()
    return any(x in t for x in ["EQH", "EQL", "SWEEP", "BSL", "SSL", "LIQUIDITY"])


def _zone_direction(z: Zone) -> Direction:
    """Определяет направление зоны"""
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
#     УЛУЧШЕННЫЙ SWING FINDER
# ---------------------------------------------------------------

def _find_swings(candles: List, left: int = 5, right: int = 2) -> List[dict]:
    """
    Находит НАСТОЯЩИЕ swing highs и swing lows.

    left=5 - свинг должен быть экстремумом среди 5 свечей слева
    right=2 - и подтверждён 2 свечами справа

    Это даёт структурные свинги, а не случайные хаи/лои.
    """
    swings = []
    n = len(candles)

    if n < left + right + 1:
        return swings

    for i in range(left, n - right):
        c = candles[i]

        # Swing High - high СТРОГО выше всех соседей слева, >= справа
        is_high = all(c.high > candles[j].high for j in range(i - left, i)) and \
                  all(c.high >= candles[j].high for j in range(i + 1, i + right + 1))

        # Swing Low - low СТРОГО ниже всех соседей слева, <= справа
        is_low = all(c.low < candles[j].low for j in range(i - left, i)) and \
                 all(c.low <= candles[j].low for j in range(i + 1, i + right + 1))

        if is_high:
            swings.append({
                "kind": "HIGH",
                "index": i,
                "price": c.high,
                "time": getattr(c, 'time', i)
            })
        if is_low:
            swings.append({
                "kind": "LOW",
                "index": i,
                "price": c.low,
                "time": getattr(c, 'time', i)
            })

    return swings


def _calculate_atr(candles: List, period: int = 14) -> float:
    """Рассчитывает ATR для buffer"""
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
#     ИСПРАВЛЕННЫЙ SL FINDER - ТОЛЬКО за swing или POI
# ---------------------------------------------------------------

def _find_structural_stop_loss(
        ctx: ChainContext,
        tf: str,
        direction: Direction,
        zone: Zone,
        entry: float
) -> Optional[float]:
    """
    ICT-логика стопа:

    1. Для OB/Breaker: стоп за границей зоны (это и есть структура)
    2. Для FVG/IDM: ищем ближайший ЗНАЧИМЫЙ swing
    3. Минимальный риск 0.5% (защита от микростопов)
    """

    DEBUG = True

    def dbg(msg):
        if DEBUG:
            print(f"      [SL] {msg}")

    candles = ctx.candles.get(tf, [])
    if not candles or len(candles) < 10:
        dbg(f"Not enough candles: {len(candles) if candles else 0}")
        return None

    dbg(f"Candles on {tf}: {len(candles)}, Entry: {entry:.5f}")
    dbg(f"Zone type: {zone.type}, Zone: {zone.low:.5f}-{zone.high:.5f}")

    # === БУФЕР ===
    atr = _calculate_atr(candles, 14)
    buffer = atr * 0.15 if atr > 0 else entry * 0.001
    dbg(f"ATR: {atr:.6f}, Buffer: {buffer:.6f}")

    # === МИНИМАЛЬНЫЙ РИСК 0.5% ===
    min_risk = entry * 0.005
    dbg(f"Min risk (0.5%): {min_risk:.5f}")

    stop_loss = None

    # === ВАРИАНТ 1: Для OB/BREAKER — стоп за границей зоны ===
    if "OB" in zone.type or "BREAKER" in zone.type:
        if direction == Direction.LONG:
            # Стоп за low зоны
            stop_loss = zone.low - buffer
            dbg(f"OB zone SL (behind zone.low): {stop_loss:.5f}")
        else:
            # Стоп за high зоны
            stop_loss = zone.high + buffer
            dbg(f"OB zone SL (behind zone.high): {stop_loss:.5f}")

        # Проверяем минимальный риск
        actual_risk = abs(entry - stop_loss)
        if actual_risk >= min_risk:
            dbg(f"✓ OB SL OK, risk: {actual_risk / entry * 100:.2f}%")
            return stop_loss
        else:
            dbg(f"OB SL too tight ({actual_risk / entry * 100:.2f}%), looking for swing...")
            stop_loss = None  # Ищем swing

    # === ВАРИАНТ 2: Ищем свинги ===
    all_swings = []

    # Текущий TF
    if len(candles) >= 6:
        swings_soft = _find_swings(candles, left=2, right=1)
        all_swings.extend(swings_soft)
    if len(candles) >= 8:
        swings_normal = _find_swings(candles, left=3, right=2)
        all_swings.extend(swings_normal)

    # Старшие TF
    for htf in ["1h", "4h"]:
        htf_candles = ctx.candles.get(htf, [])
        if htf_candles and len(htf_candles) >= 8:
            htf_swings = _find_swings(htf_candles, left=2, right=1)
            all_swings.extend(htf_swings)
            dbg(f"Swings {htf}: {len(htf_swings)}")

    # Убираем дубликаты
    unique_swings = []
    seen_prices = set()
    for s in all_swings:
        price_key = round(s["price"] / entry * 1000)
        key = (s["kind"], price_key)
        if key not in seen_prices:
            seen_prices.add(key)
            unique_swings.append(s)

    dbg(f"Total unique swings: {len(unique_swings)}")

    if direction == Direction.LONG:
        valid_lows = [s for s in unique_swings if s["kind"] == "LOW" and s["price"] < entry]
        dbg(f"Valid swing LOWs below entry: {len(valid_lows)}")

        if valid_lows:
            # Сортируем по цене (от высокой к низкой = от ближней к дальней)
            valid_lows.sort(key=lambda s: s["price"], reverse=True)

            for i, swing in enumerate(valid_lows[:5]):
                dbg(f"  Swing LOW {i}: {swing['price']:.5f}")

            # === КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ===
            # Для старших TF (1h, 4h) берём более значимый swing (2-й или 3-й)
            # Для младших (15m) берём ближайший

            if tf in ["4h", "1h"] and len(valid_lows) >= 2:
                # Пропускаем первый (мелкий) swing, берём второй
                # Но проверяем что первый действительно "мелкий" (близко к entry)
                first_swing = valid_lows[0]
                first_distance = entry - first_swing["price"]

                # Если первый swing даёт меньше 1.5% риска - пропускаем его
                if first_distance < entry * 0.015 and len(valid_lows) >= 2:
                    dbg(f"Skipping shallow swing {first_swing['price']:.5f} (only {first_distance / entry * 100:.2f}%)")
                    swing_to_use = valid_lows[1]  # Берём второй
                else:
                    swing_to_use = valid_lows[0]
            else:
                swing_to_use = valid_lows[0]

            stop_loss = swing_to_use["price"] - buffer
            dbg(f"Using swing: {swing_to_use['price']:.5f}, SL: {stop_loss:.5f}")

        # Fallback: recent low
        if stop_loss is None:
            lookback = min(50, len(candles))
            recent_low = min(c.low for c in candles[-lookback:])
            dbg(f"Recent low (last {lookback}): {recent_low:.5f}")

            if entry - recent_low >= min_risk:
                stop_loss = recent_low - buffer
                dbg(f"SL from recent low: {stop_loss:.5f}")

        # Финальные проверки
        if stop_loss is None:
            dbg("NO VALID SL FOUND!")
            return None

        if stop_loss >= entry:
            dbg(f"SL {stop_loss:.5f} >= Entry {entry:.5f}, invalid!")
            return None

        actual_risk = entry - stop_loss
        if actual_risk < min_risk:
            dbg(f"Risk {actual_risk:.5f} < min {min_risk:.5f}")
            return None

    else:  # SHORT
        valid_highs = [s for s in unique_swings if s["kind"] == "HIGH" and s["price"] > entry]
        dbg(f"Valid swing HIGHs above entry: {len(valid_highs)}")

        if valid_highs:
            valid_highs.sort(key=lambda s: s["price"])  # От низкой к высокой

            if tf in ["4h", "1h"] and len(valid_highs) >= 2:
                first_swing = valid_highs[0]
                first_distance = first_swing["price"] - entry

                if first_distance < entry * 0.015 and len(valid_highs) >= 2:
                    dbg(f"Skipping shallow swing {first_swing['price']:.5f}")
                    swing_to_use = valid_highs[1]
                else:
                    swing_to_use = valid_highs[0]
            else:
                swing_to_use = valid_highs[0]

            stop_loss = swing_to_use["price"] + buffer

        if stop_loss is None:
            lookback = min(50, len(candles))
            recent_high = max(c.high for c in candles[-lookback:])

            if recent_high - entry >= min_risk:
                stop_loss = recent_high + buffer

        if stop_loss is None:
            return None

        if stop_loss <= entry:
            return None

        actual_risk = stop_loss - entry
        if actual_risk < min_risk:
            return None

    dbg(f"✓ Final SL: {stop_loss:.5f}, Risk: {abs(entry - stop_loss) / entry * 100:.2f}%")
    return stop_loss


# ---------------------------------------------------------------
#     ИСПРАВЛЕННЫЙ TP FINDER - только на liquidity/imbalance
# ---------------------------------------------------------------

def _find_liquidity_targets(
        ctx: ChainContext,
        tf: str,
        direction: Direction,
        entry: float,
        min_rr: float = 1.5
) -> List[float]:
    """
    Находит TP на liquidity/imbalance.
    Фильтрует цели которые дают минимум min_rr.
    """
    targets = []

    # Нужен риск для расчёта минимального TP
    # Пока используем примерный 1.5% риск, потом фильтруем

    all_zones = []
    for check_tf in ["15m", "1h", "4h", "1d"]:
        det = ctx.detections.get(check_tf)
        if det and det.zones:
            for z in det.zones:
                all_zones.append((z, check_tf))

    candles = ctx.candles.get(tf, [])
    swings = _find_swings(candles, left=3, right=1) if candles else []

    for htf in ["1h", "4h"]:
        htf_candles = ctx.candles.get(htf, [])
        if htf_candles:
            htf_swings = _find_swings(htf_candles, left=3, right=1)
            swings.extend(htf_swings)

    if direction == Direction.LONG:
        # Liquidity pools выше
        liq_zones = [(z, htf) for z, htf in all_zones if _is_liquidity(z) and z.low > entry]
        for z, htf in liq_zones:
            targets.append(z.low)

        # Bearish imbalance выше (CE 50%)
        bearish_imb = [(z, htf) for z, htf in all_zones
                       if ((_is_fvg(z) and "DOWN" in z.type.upper()) or
                           (_is_ob(z) and "BEAR" in z.type.upper()))
                       and z.low > entry]
        for z, htf in bearish_imb:
            targets.append((z.low + z.high) / 2)

        # Swing highs
        swing_highs = [s["price"] for s in swings if s["kind"] == "HIGH" and s["price"] > entry]
        targets.extend(swing_highs)

        # Сортируем и убираем дубликаты
        targets = sorted(set(targets))

    else:  # SHORT
        liq_zones = [(z, htf) for z, htf in all_zones if _is_liquidity(z) and z.high < entry]
        for z, htf in liq_zones:
            targets.append(z.high)

        bullish_imb = [(z, htf) for z, htf in all_zones
                       if ((_is_fvg(z) and "UP" in z.type.upper()) or
                           (_is_ob(z) and "BULL" in z.type.upper()))
                       and z.high < entry]
        for z, htf in bullish_imb:
            targets.append((z.low + z.high) / 2)

        swing_lows = [s["price"] for s in swings if s["kind"] == "LOW" and s["price"] < entry]
        targets.extend(swing_lows)

        targets = sorted(set(targets), reverse=True)

    return targets[:10]  # Возвращаем больше целей для фильтрации


# ---------------------------------------------------------------
#     Price in zone checker
# ---------------------------------------------------------------

def _is_price_in_zone(ctx: ChainContext, zone: Zone, tf: str) -> bool:
    """Проверяет, находится ли текущая цена близко к зоне"""
    candles = ctx.candles.get(tf, [])
    if not candles:
        return False

    current_price = candles[-1].close
    zone_size = zone.high - zone.low

    # Буфер = 50% от размера зоны или 1.5% от цены (что больше)
    buffer = max(zone_size * 0.5, current_price * 0.015)

    expanded_low = zone.low - buffer
    expanded_high = zone.high + buffer

    return expanded_low <= current_price <= expanded_high


# ---------------------------------------------------------------
#     ИСПРАВЛЕННЫЙ SIGNAL BUILDER
# ---------------------------------------------------------------

def _make_signal(
        ctx: ChainContext,
        chain_id: str,
        direction: Direction,
        zone: Zone,
        tf: Optional[str] = None,
        description: str = "",
) -> Optional[ChainSignal]:
    """
    ИСПРАВЛЕННОЕ построение сигнала:
    - SL ТОЛЬКО за swing (без fallback на zone!)
    - TP ТОЛЬКО на liquidity или imbalance
    """
    tf = tf or zone.tf

    # DEBUG - включить для отладки
    DEBUG = True

    def dbg(msg):
        if DEBUG:
            print(f"    [DBG {chain_id}] {msg}")

    # === BIAS FILTERING ===
    if ctx.bias_contexts and tf in ctx.bias_contexts:
        bias = ctx.bias_contexts[tf]
        if bias.bias == "STRONG_BEARISH" and direction == Direction.LONG:
            dbg(f"REJECTED: STRONG_BEARISH bias, direction LONG")
            return None
        if bias.bias == "STRONG_BULLISH" and direction == Direction.SHORT:
            dbg(f"REJECTED: STRONG_BULLISH bias, direction SHORT")
            return None

    # Проверяем, что цена близко к зоне
    if not _is_price_in_zone(ctx, zone, tf):
        dbg(f"REJECTED: price not in zone {zone.low:.5f}-{zone.high:.5f}")
        return None

    low, high = zone.low, zone.high
    dbg(f"Zone: {zone.type} @ {low:.5f}-{high:.5f}")

    # === ENTRY на основе типа зоны ===
    if "OB" in zone.type:
        if direction == Direction.LONG:
            entry = low + (high - low) * 0.62
        else:
            entry = high - (high - low) * 0.62
    elif "FVG" in zone.type:
        entry = (low + high) / 2.0
    elif "MB" in zone.type:
        if direction == Direction.LONG:
            entry = low + (high - low) * 0.3
        else:
            entry = high - (high - low) * 0.3
    elif "BREAKER" in zone.type:
        entry = low + (high - low) * 0.62
    elif "IDM" in zone.type:
        if direction == Direction.LONG:
            entry = low + (high - low) * 0.7
        else:
            entry = high - (high - low) * 0.7
    else:
        entry = (low + high) / 2.0

    dbg(f"Entry: {entry:.5f}, Direction: {direction}")

    # === STOP LOSS - СТРУКТУРНЫЙ ===
    sl = _find_structural_stop_loss(ctx, tf, direction, zone, entry)

    if sl is None:
        dbg(f"REJECTED: no structural SL found")
        return None

    risk = abs(entry - sl)
    risk_percent = risk / entry * 100

    dbg(f"SL: {sl:.5f}, Risk: {risk_percent:.2f}%")

    if risk <= 0:
        dbg(f"REJECTED: risk <= 0")
        return None

    # === TAKE PROFITS - на LIQUIDITY/IMBALANCE ===
    targets = _find_liquidity_targets(ctx, tf, direction, entry)
    dbg(f"Raw targets: {[f'{t:.5f}' for t in targets[:5]]}")

    # Фильтруем цели: оставляем только те что дают RR >= 1.5
    min_tp_distance = risk * 1.5  # Минимум RR 1.5

    if direction == Direction.LONG:
        valid_targets = [t for t in targets if t - entry >= min_tp_distance]
    else:
        valid_targets = [t for t in targets if entry - t >= min_tp_distance]

    dbg(f"Valid targets (RR>=1.5): {[f'{t:.5f}' for t in valid_targets[:3]]}")

    if valid_targets:
        tp1 = valid_targets[0]
        tp2 = valid_targets[1] if len(valid_targets) > 1 else (
            tp1 + risk * 1.0 if direction == Direction.LONG else tp1 - risk * 1.0
        )
        dbg(f"TP from liquidity: {tp1:.5f}, {tp2:.5f}")
    else:
        # Fallback: RR мультипликаторы (если нет подходящей ликвидности)
        if direction == Direction.LONG:
            tp1 = entry + risk * 2.0
            tp2 = entry + risk * 3.5
        else:
            tp1 = entry - risk * 2.0
            tp2 = entry - risk * 3.5
        dbg(f"TP fallback (no valid liquidity): {tp1:.5f}, {tp2:.5f}")

    # === RR CALCULATION ===
    rr = abs(tp1 - entry) / risk if risk > 0 else 0.0
    dbg(f"RR: {rr:.2f}")

    if rr < 1.5:
        dbg(f"REJECTED: RR {rr:.2f} < 1.5")
        return None

    # === FINAL VALIDATION ===
    if direction == Direction.LONG:
        if tp1 <= entry or sl >= entry:
            dbg(f"REJECTED: invalid levels for LONG")
            return None
    else:
        if tp1 >= entry or sl <= entry:
            dbg(f"REJECTED: invalid levels for SHORT")
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
        description=f"{description} | SL: Structural | TP: Liquidity",
    )


# ================================================================
#               ЦЕПОЧКИ
# ================================================================

# ---------------------------------------------------------------
# CHAIN 1.1
# ---------------------------------------------------------------

class Chain_1_1:
    """
    V1(X): VC.D (OB,FVG)
    V2(X-1): VC.4h (OB,FVG) в V1
    V3(X-2): VC.1h (OB,FVG)
    V4(X-3): VC.15m (FVG) в V3
    """
    chain_id = "1.1"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Daily OB/FVG
        zone_d = _get_primary_zone(ctx.detections.get("1d"), ["OB", "FVG"])
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        # V2: 4h zone ВНУТРИ V1
        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if (_is_ob(z) or _is_fvg(z)) and _is_zone_inside(z, zone_d):
                zone_4h = z
                break

        if not zone_4h:
            return []

        # V3: 1h OB/FVG
        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        # V4: 15m FVG ВНУТРИ V3
        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break

        if not zone_15m:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_15m,
            tf="15m",
            description="Chain 1.1: D->4h(in)->1h->15m(in)",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 1.2 - IDM-driven
# ---------------------------------------------------------------

class Chain_1_2:
    """
    V1(X): VC.D (IDM)
    V2(X-1): VC.4h (IDM)
    V3(X-2): VC.1h (OB,FVG)
    V4(X-3): VC.15m (FVG) в V3
    """
    chain_id = "1.2"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Daily IDM
        zone_d_idm = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_idm(z):
                zone_d_idm = z
                break

        if not zone_d_idm:
            return []

        direction = _zone_direction(zone_d_idm)

        # V2: 4h IDM
        zone_4h_idm = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_idm(z):
                zone_4h_idm = z
                break

        if not zone_4h_idm:
            return []

        # V3: 1h OB/FVG
        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        # V4: 15m FVG внутри V3
        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break

        if not zone_15m:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_15m,
            tf="15m",
            description="Chain 1.2: D(IDM)->4h(IDM)->1h->15m(in)",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 1.3
# ---------------------------------------------------------------

class Chain_1_3:
    """
    V1(X): VC.D (OB,FVG)
    V2(X-1): VC.4h (IDM)
    V3(X-2): VC.1h (OB,FVG)
    V4(X-3): VC.15m (FVG) в V3
    """
    chain_id = "1.3"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Daily OB/FVG
        zone_d = _get_primary_zone(ctx.detections.get("1d"), ["OB", "FVG"])
        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        # V2: 4h IDM (требуется)
        idm_4h = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_idm(z):
                idm_4h = z
                break

        if not idm_4h:
            return []

        # V3: 1h OB/FVG
        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        # V4: 15m FVG внутри V3
        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break

        if not zone_15m:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_15m,
            tf="15m",
            description="Chain 1.3: D->4h(IDM)->1h->15m(in)",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 1.4
# ---------------------------------------------------------------

class Chain_1_4:
    """
    V1(X): VC.D (FVG)
    V2(X-1): VC.4h (OB,FVG) в V1
    После FF FVG V1:
    V3(X-2): VC.1h (OB,FVG)
    V4(X-3): VC.15m (FVG) в V3
    """
    chain_id = "1.4"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Daily FVG (только FVG!)
        zone_d = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_d = z
                break

        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        # V2: 4h zone внутри Daily FVG
        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if (_is_ob(z) or _is_fvg(z)) and _is_zone_inside(z, zone_d):
                zone_4h = z
                break

        if not zone_4h:
            return []

        # V3: 1h OB/FVG
        zone_1h = _get_primary_zone(ctx.detections.get("1h"), ["OB", "FVG"])
        if not zone_1h:
            return []

        # V4: 15m FVG внутри V3
        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None
        for z in zones_15m:
            if _is_fvg(z) and _is_zone_inside(z, zone_1h):
                zone_15m = z
                break

        if not zone_15m:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_15m,
            tf="15m",
            description="Chain 1.4: D(FVG)->4h(in)->1h->15m(in)",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 1.5
# ---------------------------------------------------------------

class Chain_1_5:
    """
    V1(X): VC.D (FVG)
    V2(X-1): FVG реакция - возврат в FVG
    V3(X-2): VC.4h (FVG) внутри Daily
    """
    chain_id = "1.5"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Daily FVG
        zone_d = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_d = z
                break

        if not zone_d:
            return []

        direction = _zone_direction(zone_d)

        # Проверяем, что цена вернулась в Daily FVG (реакция)
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

        # V3: 4h FVG внутри Daily FVG
        zones_4h = ctx.detections.get("4h", DetectionResult([], None)).zones
        zone_4h = None
        for z in zones_4h:
            if _is_fvg(z) and _is_zone_inside(z, zone_d, tolerance=0.2):
                zone_4h = z
                break

        if not zone_4h:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_4h,
            tf="4h",
            description="Chain 1.5: D(FVG)->reaction->4h(FVG)",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 2.6
# ---------------------------------------------------------------

class Chain_2_6:
    """
    V1(X): Снятие ликвидности FH.4h;
    Закрыть период 4h до фрактала;
    Получить зону POI по тени свечи;
    Образовать новый фрактал.
    V2(X-1): Возврат в зону POI 4h
    V3(X-2): FVG.15m|30m|45m частично или полностью в V2
    """
    chain_id = "2.6"

    def _get_poi_from_fractal_candle(self, candles: List, fractal_idx: int) -> Optional[Zone]:
        """Получает POI зону по тени свечи фрактала"""
        if fractal_idx >= len(candles):
            return None

        c = candles[fractal_idx]

        # POI = тень свечи
        if c.close > c.open:  # Бычья свеча
            # POI = верхняя тень
            poi_low = max(c.open, c.close)
            poi_high = c.high
        else:  # Медвежья свеча
            # POI = нижняя тень
            poi_low = c.low
            poi_high = min(c.open, c.close)

        if poi_high <= poi_low:
            return None

        return Zone(
            tf="4h",
            low=poi_low,
            high=poi_high,
            type="POI"
        )

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: Ищем FH на 4h (fractal high/low)
        fh_zones = [z for z in ctx.detections.get("4h", DetectionResult([], None)).zones if _is_fh(z)]
        if not fh_zones:
            return []

        # Берём последний FH
        fh_zone = fh_zones[-1]
        direction = Direction.SHORT if "HIGH" in fh_zone.type.upper() else Direction.LONG

        # Находим индекс свечи, которая создала фрактал
        candles_4h = ctx.candles.get("4h", [])
        if not candles_4h:
            return []

        fractal_idx = None
        for i in range(len(candles_4h) - 1, max(0, len(candles_4h) - 20), -1):
            c = candles_4h[i]
            if "HIGH" in fh_zone.type:
                if c.high >= fh_zone.low and c.high <= fh_zone.high:
                    fractal_idx = i
                    break
            else:
                if c.low >= fh_zone.low and c.low <= fh_zone.high:
                    fractal_idx = i
                    break

        if fractal_idx is None:
            return []

        # Получаем POI по тени свечи
        poi_zone = self._get_poi_from_fractal_candle(candles_4h, fractal_idx)
        if not poi_zone:
            return []

        # V2: Проверяем возврат в POI
        current_4h = candles_4h[-1]
        poi_touched = False

        # Проверяем последние 5 свечей на касание POI
        for c in candles_4h[-5:]:
            if c.low <= poi_zone.high and c.high >= poi_zone.low:
                poi_touched = True
                break

        if not poi_touched:
            return []

        # V3: Ищем FVG на 15m в/около POI
        zones_15m = ctx.detections.get("15m", DetectionResult([], None)).zones
        zone_15m = None

        for z in zones_15m:
            if _is_fvg(z):
                # Проверяем направление FVG
                if _zone_direction(z) != direction:
                    continue
                # Проверяем пересечение с POI
                if _zones_overlap(z, poi_zone):
                    zone_15m = z
                    break

        if not zone_15m:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_15m,
            tf="15m",
            description="Chain 2.6: Liquidity sweep->POI->15m FVG",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# CHAIN 3.2
# ---------------------------------------------------------------

class Chain_3_2:
    """
    V1(X): FVG.4h (первый заход)
    4h закрылась выше - отмена
    V2(X-1): образовать FVG.1h в свече 4h −1 или последующей 4h
    """
    chain_id = "3.2"

    def __init__(self):
        # Трекинг посещенных зон (для проверки "первого захода")
        self.visited_zones = {}  # key -> timestamp

    def _get_zone_key(self, zone: Zone) -> str:
        """Уникальный ключ для зоны"""
        return f"{zone.tf}_{zone.type}_{zone.low:.2f}_{zone.high:.2f}"

    def _is_first_touch(self, zone: Zone, candles: List) -> bool:
        """Проверяет, является ли это первым заходом в зону"""
        zone_key = self._get_zone_key(zone)

        # Если зона уже была посещена - это не первый заход
        if zone_key in self.visited_zones:
            return False

        # Находим, когда зона появилась
        # Упрощённо: считаем, что зона "новая" если появилась в последних 10 свечах
        zone_age = 0
        for i in range(len(candles) - 1, max(0, len(candles) - 10), -1):
            c = candles[i]
            if c.low <= zone.high and c.high >= zone.low:
                zone_age += 1

        # Если зону трогали больше 1 раза - это не первый заход
        return zone_age <= 1

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: FVG на 4h
        zone_4h = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_fvg(z):
                zone_4h = z
                break

        if not zone_4h:
            return []

        direction = _zone_direction(zone_4h)
        candles_4h = ctx.candles.get("4h", [])

        if not candles_4h:
            return []

        # Проверяем первый заход
        if not self._is_first_touch(zone_4h, candles_4h):
            return []

        # Проверяем, что свеча вошла в FVG
        last = candles_4h[-1]
        entered = False

        if direction == Direction.LONG:
            # Для лонга: свеча должна коснуться верхней части FVG
            if last.low <= zone_4h.high:
                entered = True
                # Проверяем, не закрылась ли выше (отмена)
                if last.close > zone_4h.high:
                    return []
        else:  # SHORT
            # Для шорта: свеча должна коснуться нижней части FVG
            if last.high >= zone_4h.low:
                entered = True
                # Проверяем, не закрылась ли ниже (отмена)
                if last.close < zone_4h.low:
                    return []

        if not entered:
            return []

        # Отмечаем зону как посещённую
        zone_key = self._get_zone_key(zone_4h)
        self.visited_zones[zone_key] = candles_4h[-1].time if hasattr(candles_4h[-1], 'time') else 0

        # V2: FVG на 1h в направлении движения
        zone_1h = None
        for z in ctx.detections.get("1h", DetectionResult([], None)).zones:
            if _is_fvg(z) and _zone_direction(z) == direction:
                zone_1h = z
                break

        if not zone_1h:
            return []

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_1h,
            tf="1h",
            description="Chain 3.2: First 4h FVG touch->1h continuation",
        )

        return [sig] if sig else []


# ---------------------------------------------------------------
# Signal_1
# ---------------------------------------------------------------

class Signal_1:
    """
    V1(X): FH.D
    V2(X-1): OB.4h
    """
    chain_id = "Signal_1"

    async def analyze(self, ctx: ChainContext) -> List[ChainSignal]:
        # V1: FH на дневном
        zone_fh = None
        for z in ctx.detections.get("1d", DetectionResult([], None)).zones:
            if _is_fh(z):
                zone_fh = z
                break

        if not zone_fh:
            return []

        # V2: OB на 4h
        zone_ob = None
        for z in ctx.detections.get("4h", DetectionResult([], None)).zones:
            if _is_ob(z):
                zone_ob = z
                break

        if not zone_ob:
            return []

        direction = _zone_direction(zone_ob)

        sig = _make_signal(
            ctx=ctx,
            chain_id=self.chain_id,
            direction=direction,
            zone=zone_ob,
            tf="4h",
            description="Signal_1: FH.D + OB.4h",
        )

        return [sig] if sig else []

# END