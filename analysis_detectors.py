# analysis_detectors.py (ИСПРАВЛЕННАЯ ВЕРСИЯ С TIMESTAMPS)
# =========================================================

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from analysis_interfaces import Zone, DetectionResult, VolumeContext
from data.data_interfaces import Candle


# -------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ОБЩИЕ)
# -------------------------------------

def _get_candle_timestamp(candle) -> Optional[int]:
    """
    Безопасно извлекает timestamp из свечи
    """
    # Пробуем разные варианты получения времени
    ts = getattr(candle, 'time', None)
    if ts is None:
        ts = getattr(candle, 'timestamp', None)

    # Если это datetime объект
    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1000)

    # Если это уже число
    if isinstance(ts, (int, float)):
        # Проверяем, не в секундах ли (слишком маленькое число)
        if ts < 1000000000000:  # Меньше чем timestamp в миллисекундах
            return int(ts * 1000)
        return int(ts)

    # Fallback на текущее время
    import time
    return int(time.time() * 1000)


def _avg_body(candles: List[Candle], lookback: int = 20) -> float:
    """
    Средний размер тела свечи за lookback.
    """
    if not candles:
        return 0.0

    sub = candles[-lookback:] if len(candles) > lookback else candles
    bodies = [abs(c.close - c.open) for c in sub]
    if not bodies:
        return 0.0

    return sum(bodies) / len(bodies)


def _atr(candles: List[Candle], period: int = 14) -> float:
    """True Range / ATR для оценки силы движения и фильтра свингов."""
    if len(candles) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    use = trs[-period:]
    return sum(use) / len(use)


def _find_swings(candles: List[Candle], left: int = 2, right: int = 2):
    """
    Ищем фрактальные swing highs / swing lows:
    HIGH – выше N слева и N справа
    LOW  – ниже N слева и N справа
    """
    swings = []
    n = len(candles)
    for i in range(left, n - right):
        high = candles[i].high
        low = candles[i].low

        is_high = all(high > candles[i - k].high for k in range(1, left + 1)) and \
                  all(high > candles[i + k].high for k in range(1, right + 1))
        is_low = all(low < candles[i - k].low for k in range(1, left + 1)) and \
                 all(low < candles[i + k].low for k in range(1, right + 1))

        if is_high:
            swings.append({"kind": "HIGH", "index": i, "price": high})
        if is_low:
            swings.append({"kind": "LOW", "index": i, "price": low})

    swings.sort(key=lambda s: s["index"])
    return swings


def _filter_major_swings(swings, atr: float, min_factor: float = 0.5):
    """
    Оставляем только значимые свинги:
    амплитуда > min_factor * ATR.
    Это external structure относительно мелких дерганий.
    """
    if atr <= 0 or len(swings) < 2:
        return swings

    majors = [swings[0]]
    for s in swings[1:]:
        prev = majors[-1]
        if abs(s["price"] - prev["price"]) >= atr * min_factor:
            majors.append(s)
    return majors


def _classify_structure(swings):
    """
    Строим базовую структуру HH/HL/LH/LL и общий тренд:
    BULLISH / BEARISH / RANGE.
    """
    if len(swings) < 3:
        return "RANGE", []

    highs = [s for s in swings if s["kind"] == "HIGH"]
    lows = [s for s in swings if s["kind"] == "LOW"]

    labels = []
    structure = "RANGE"

    if len(highs) >= 2:
        if highs[-1]["price"] > highs[-2]["price"]:
            structure = "BULLISH"
            labels.append("HH")
        elif highs[-1]["price"] < highs[-2]["price"]:
            structure = "BEARISH"
            labels.append("LH")

    if len(lows) >= 2:
        if lows[-1]["price"] > lows[-2]["price"]:
            labels.append("HL")
            if structure == "BEARISH":
                structure = "RANGE"
        elif lows[-1]["price"] < lows[-2]["price"]:
            labels.append("LL")
            if structure == "BULLISH":
                structure = "RANGE"

    return structure, labels


def _find_bos(swings, candles: List[Candle]):
    """
    Ищем направление последнего BOS:
    UP / DOWN / NONE
    """
    if len(swings) < 3:
        return "NONE"

    last_close = candles[-1].close
    a, b, c = swings[-3], swings[-2], swings[-1]

    direction = "NONE"
    if c["kind"] == "HIGH" and last_close > b["price"]:
        direction = "UP"
    if c["kind"] == "LOW" and last_close < b["price"]:
        direction = "DOWN"

    return direction


def _protected_extremes(swings, candles: List[Candle]):
    """
    Protected High/Low:
    последний swing, за который цена ещё не закрывалась.
    """
    last_index = len(candles) - 1
    prot_high = None
    prot_low = None

    # Protected High
    for s in reversed(swings):
        if s["kind"] != "HIGH":
            continue
        broken = any(candles[i].close > s["price"] for i in range(s["index"] + 1, last_index + 1))
        if not broken:
            prot_high = s["price"]
            break

    # Protected Low
    for s in reversed(swings):
        if s["kind"] != "LOW":
            continue
        broken = any(candles[i].close < s["price"] for i in range(s["index"] + 1, last_index + 1))
        if not broken:
            prot_low = s["price"]
            break

    return prot_high, prot_low


def _liquidity_target(swings, bias: str, current_price: float):
    """
    Простая модель liquidity draw:
    - BULLISH → ближайший high выше цены
    - BEARISH → ближайший low ниже цены
    """
    target = None
    target_kind = None

    if bias == "BULLISH":
        candidates = [s for s in swings if s["kind"] == "HIGH" and s["price"] > current_price]
        if candidates:
            target = min(candidates, key=lambda s: s["price"])
            target_kind = "HIGH"

    elif bias == "BEARISH":
        candidates = [s for s in swings if s["kind"] == "LOW" and s["price"] < current_price]
        if candidates:
            target = max(candidates, key=lambda s: s["price"])
            target_kind = "LOW"

    if not target:
        return None

    return f"{target_kind}@{round(target['price'], 2)}"


def detect_market_structure(candles: List[Candle], tf: str) -> VolumeContext:
    """
    FULL ICT-style market structure:
    - swings (HH/HL/LH/LL)
    - BOS / CHoCH
    - protected highs/lows
    - liquidity target
    - bias (BULLISH / BEARISH / RANGE)
    """
    if len(candles) < 10:
        return VolumeContext(
            tf=tf,
            bias="RANGE",
            structure="",
            note="not_enough_candles",
            killzone="None",
        )

    atr = _atr(candles, period=14)
    swings = _find_swings(candles, left=2, right=2)
    if not swings:
        return VolumeContext(
            tf=tf,
            bias="RANGE",
            structure="",
            note="no_swings",
            killzone="None",
        )

    major_swings = _filter_major_swings(swings, atr, min_factor=0.5)
    base_for_structure = major_swings if len(major_swings) >= 3 else swings

    structure, labels = _classify_structure(base_for_structure)
    bias = structure  # BULLISH / BEARISH / RANGE

    bos_dir = _find_bos(base_for_structure, candles)

    # CHoCH – грубо: BOS против текущей структуры
    choch = "NONE"
    if bos_dir == "UP" and structure == "BEARISH":
        choch = "BULLISH_CH"
    elif bos_dir == "DOWN" and structure == "BULLISH":
        choch = "BEARISH_CH"

    prot_high, prot_low = _protected_extremes(base_for_structure, candles)
    last_close = candles[-1].close
    liq = _liquidity_target(base_for_structure, bias, last_close)

    note_parts = [
        f"bos={bos_dir}",
        f"choch={choch}",
        f"prot_high={round(prot_high, 2) if prot_high is not None else 'None'}",
        f"prot_low={round(prot_low, 2) if prot_low is not None else 'None'}",
        f"liq_target={liq or 'None'}",
        f"atr={round(atr, 2)}",
    ]
    note = "; ".join(note_parts)

    return VolumeContext(
        tf=tf,
        bias=bias,
        structure="|".join(labels),
        note=note,
        killzone="None",
    )


def _find_swing_high_idx(candles: List[Candle], lookback: int = 20) -> Optional[int]:
    """
    Swing High:
        high[i] > high[i-1] AND high[i] > high[i+1]
    """
    if len(candles) < 5:
        return None

    start = max(2, len(candles) - lookback)
    for i in range(len(candles) - 2, start - 1, -1):
        c_prev = candles[i - 1]
        c = candles[i]
        c_next = candles[i + 1]

        if c.high > c_prev.high and c.high > c_next.high:
            return i

    return None


def _find_swing_low_idx(candles: List[Candle], lookback: int = 20) -> Optional[int]:
    """
    Swing Low:
        low[i] < low[i-1] AND low[i] < low[i+1]
    """
    if len(candles) < 5:
        return None

    start = max(2, len(candles) - lookback)
    for i in range(len(candles) - 2, start - 1, -1):
        c_prev = candles[i - 1]
        c = candles[i]
        c_next = candles[i + 1]

        if c.low < c_prev.low and c.low < c_next.low:
            return i

    return None


# ==========================================================
#                 ORDER BLOCK DETECTOR (С TIMESTAMPS)
# ==========================================================

class OrderBlockDetector:
    """
    Упрощённый ICT-style Order Block с добавлением timestamps.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.min_candles: int = config.get("min_candles", 30)
        self.displacement_factor: float = config.get("displacement_factor", 1.2)
        self.lookahead: int = config.get("lookahead", 4)

    def _detect_bullish(self, candles: List[Candle], tf: str) -> Optional[Zone]:
        if len(candles) < self.min_candles:
            return None

        avg_body = _avg_body(candles, lookback=50)
        if avg_body <= 0:
            return None

        # идём с конца, ищем последнюю подходящую медвежью базу
        for i in range(len(candles) - 3, 2, -1):
            base = candles[i]
            # база должна быть медвежья
            if base.close >= base.open:
                continue

            # в ближайших N свечах ищем сильную бычью свечу
            found_impulse = False
            for j in range(i + 1, min(len(candles), i + 1 + self.lookahead)):
                imp = candles[j]
                if imp.close <= imp.open:
                    continue
                body = abs(imp.close - imp.open)
                if body >= avg_body * self.displacement_factor:
                    found_impulse = True
                    break

            if not found_impulse:
                continue

            # нашли Bullish OB
            low = base.low
            high = base.open

            # ИСПРАВЛЕНО: Получаем timestamp
            timestamp = _get_candle_timestamp(base)

            return Zone(
                tf=tf,
                high=high,
                low=low,
                type="OB_BULL",
                timestamp=timestamp,  # ДОБАВЛЕНО
                candle_index=i  # ДОБАВЛЕНО
            )

        return None

    def _detect_bearish(self, candles: List[Candle], tf: str) -> Optional[Zone]:
        if len(candles) < self.min_candles:
            return None

        avg_body = _avg_body(candles, lookback=50)
        if avg_body <= 0:
            return None

        for i in range(len(candles) - 3, 2, -1):
            base = candles[i]
            # база должна быть бычья
            if base.close <= base.open:
                continue

            found_impulse = False
            for j in range(i + 1, min(len(candles), i + 1 + self.lookahead)):
                imp = candles[j]
                if imp.close >= imp.open:
                    continue
                body = abs(imp.close - imp.open)
                if body >= avg_body * self.displacement_factor:
                    found_impulse = True
                    break

            if not found_impulse:
                continue

            high = base.high
            low = base.open

            # ИСПРАВЛЕНО: Получаем timestamp
            timestamp = _get_candle_timestamp(base)

            return Zone(
                tf=tf,
                high=high,
                low=low,
                type="OB_BEAR",
                timestamp=timestamp,  # ДОБАВЛЕНО
                candle_index=i  # ДОБАВЛЕНО
            )

        return None

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        zones: List[Zone] = []

        bull = self._detect_bullish(candles, tf)
        if bull:
            zones.append(bull)

        bear = self._detect_bearish(candles, tf)
        if bear:
            zones.append(bear)

        return DetectionResult(zones=zones, context=None)


# ==========================================================
#                 FAIR VALUE GAP DETECTOR (С TIMESTAMPS)
# ==========================================================

class FairValueGapDetector:
    """
    ICT-style FVG детектор с добавлением timestamps.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.min_candles: int = config.get("min_candles", 30)
        self.min_size_factor: float = config.get("min_size_factor", 0.25)
        self.max_fvg_count: int = config.get("max_fvg_count", 3)

    def _avg_range(self, candles: List[Candle], lookback: int = 50) -> float:
        """
        Средний диапазон свечи (high-low) за lookback баров.
        """
        if not candles:
            return 0.0

        sub = candles[-lookback:] if len(candles) > lookback else candles
        ranges = [c.high - c.low for c in sub if (c.high - c.low) > 0]
        if not ranges:
            return 0.0

        return sum(ranges) / len(ranges)

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        zones: List[Zone] = []

        if len(candles) < 3 or len(candles) < self.min_candles:
            return DetectionResult(zones=zones, context=None)

        avg_range = self._avg_range(candles, lookback=50)
        if avg_range <= 0:
            return DetectionResult(zones=zones, context=None)

        min_gap = avg_range * self.min_size_factor

        # Идём с конца, чтобы найти самые свежие FVG
        for i in range(len(candles) - 1, 1, -1):
            if len(zones) >= self.max_fvg_count:
                break

            c0 = candles[i - 2]
            c1 = candles[i - 1]
            c2 = candles[i]

            # ИСПРАВЛЕНО: Получаем timestamp от свечи c2
            timestamp = _get_candle_timestamp(c2)

            # Bullish FVG (овербаланс вверх)
            if c0.high < c2.low:
                gap_low = c0.high
                gap_high = c2.low
                size = gap_high - gap_low
                if size >= min_gap:
                    zones.append(
                        Zone(
                            tf=tf,
                            high=gap_high,
                            low=gap_low,
                            type="FVG_UP",
                            timestamp=timestamp,  # ДОБАВЛЕНО
                            candle_index=i  # ДОБАВЛЕНО
                        )
                    )

            # Bearish FVG (овербаланс вниз)
            if c0.low > c2.high:
                gap_low = c2.high
                gap_high = c0.low
                size = gap_high - gap_low
                if size >= min_gap:
                    zones.append(
                        Zone(
                            tf=tf,
                            high=gap_high,
                            low=gap_low,
                            type="FVG_DOWN",
                            timestamp=timestamp,  # ДОБАВЛЕНО
                            candle_index=i  # ДОБАВЛЕНО
                        )
                    )

        # Сортируем по времени (самые новые первые)
        zones.sort(key=lambda z: z.candle_index if z.candle_index else 0, reverse=True)

        return DetectionResult(zones=zones, context=None)


# ==========================================================
#                 FRACTAL DETECTOR (С TIMESTAMPS)
# ==========================================================

class FractalDetector:
    """
    Fractal / FH Detector для цепочки 2.6 с добавлением timestamps
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.left = config.get("left", 2)
        self.right = config.get("right", 2)
        self.lookback = config.get("lookback", 150)
        self.detect_highs = config.get("detect_highs", True)
        self.detect_lows = config.get("detect_lows", True)

    def _is_fractal_high(self, candles: List[Candle], i: int) -> bool:
        """
        Простой swing high:
            high[i] > high[i-k..i-1] и high[i] > high[i+1..i+k]
        """
        c = candles[i]
        left_slice = candles[i - self.left: i]
        right_slice = candles[i + 1: i + 1 + self.right]

        if not left_slice or not right_slice:
            return False

        h = c.high
        if any(h <= x.high for x in left_slice):
            return False
        if any(h <= x.high for x in right_slice):
            return False

        return True

    def _is_fractal_low(self, candles: List[Candle], i: int) -> bool:
        """
        Простой swing low:
            low[i] < low[i-k..i-1] и low[i] < low[i+1..i+k]
        """
        c = candles[i]
        left_slice = candles[i - self.left: i]
        right_slice = candles[i + 1: i + 1 + self.right]

        if not left_slice or not right_slice:
            return False

        l = c.low
        if any(l >= x.low for x in left_slice):
            return False
        if any(l >= x.low for x in right_slice):
            return False

        return True

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        zones: List[Zone] = []

        n = len(candles)
        if n < (self.left + self.right + 1):
            return DetectionResult(zones=zones, context=None)

        start = max(self.left, n - self.lookback)
        end = n - self.right

        for i in range(start, end):
            c = candles[i]

            # ИСПРАВЛЕНО: Получаем timestamp
            timestamp = _get_candle_timestamp(c)

            # Fractal High → зона по верхней тени
            if self.detect_highs and self._is_fractal_high(candles, i):
                body_high = max(c.open, c.close)
                zone_high = c.high
                zone_low = body_high

                zones.append(
                    Zone(
                        tf=tf,
                        high=zone_high,
                        low=zone_low,
                        type="FH_HIGH",
                        timestamp=timestamp,  # ДОБАВЛЕНО
                        candle_index=i  # ДОБАВЛЕНО
                    )
                )

            # Fractal Low → зона по нижней тени
            if self.detect_lows and self._is_fractal_low(candles, i):
                body_low = min(c.open, c.close)
                zone_low = c.low
                zone_high = body_low

                zones.append(
                    Zone(
                        tf=tf,
                        high=zone_high,
                        low=zone_low,
                        type="FH_LOW",
                        timestamp=timestamp,  # ДОБАВЛЕНО
                        candle_index=i  # ДОБАВЛЕНО
                    )
                )

        return DetectionResult(zones=zones, context=None)


# ==========================================================
#      VOLUMECONTEXT BUILDER → ICT BIAS + STRUCTURE
# ==========================================================

class VolumeContextBuilder:
    """
    Строит полный контекст:
    - bias (BULLISH / BEARISH / RANGE)
    - структура (HH/HL/LH/LL)
    - BOS / CHoCH
    - protected highs / lows
    - liquidity target
    """

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        if not candles or len(candles) < 10:
            ctx = VolumeContext(
                tf=tf,
                bias="RANGE",
                structure="",
                note="not_enough_candles",
                killzone="None",
            )
            return DetectionResult([], ctx)

        ctx = detect_market_structure(candles, tf)
        return DetectionResult([], ctx)