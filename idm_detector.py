# idm_detector.py
# ИСПРАВЛЕНО: Убрано дублирование _get_candle_timestamp, используется utils

from dataclasses import dataclass
from typing import List, Optional

from analysis_interfaces import Zone, DetectionResult
from utils import get_candle_timestamp, avg_body
from data.data_interfaces import Candle


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # "HIGH" или "LOW"


class IDMDetector:
    """
    ICT-style Inducement Detector (IDM) v2

    Что делает:
    - Ищет свинги (high/low) по локальным экстремумам
    - Ищет equal-high / equal-low зоны (ликвидность)
    - Ищет свип этой ликвидности хвостом (wick sweep)
    - Ищет микро-displacement (сильную свечу в обратную сторону)
    - Возвращает зоны:
        • IDM_BEAR (inducement в лонг, реальный шорт)
        • IDM_BULL (inducement в шорт, реальный лонг)
    """

    def __init__(self,
                 eq_tolerance: float = 0.0015,    # 0.15% разница для equal highs/lows
                 sweep_min: float = 0.0003,       # 0.03% минимальный прокол над уровнем
                 sweep_max: float = 0.003,        # 0.3% максимальный прокол
                 lookback_swings: int = 150,      # сколько последних свечей сканим
                 max_lookahead_sweep: int = 10,   # насколько далеко вперёд ищем свип после уровня
                 max_lookahead_displacement: int = 5,
                 min_body_factor: float = 1.5):   # тело displacement-свечи относительно средней
        self.eq_tolerance = eq_tolerance
        self.sweep_min = sweep_min
        self.sweep_max = sweep_max
        self.lookback_swings = lookback_swings
        self.max_lookahead_sweep = max_lookahead_sweep
        self.max_lookahead_displacement = max_lookahead_displacement
        self.min_body_factor = min_body_factor

    def _find_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """
        Простейшие свинги: high[i] > high[i±1], low[i] < low[i±1]
        """
        swings: List[SwingPoint] = []
        n = len(candles)
        if n < 3:
            return swings

        start = max(1, n - self.lookback_swings)
        for i in range(start, n - 1):
            prev_c = candles[i - 1]
            c = candles[i]
            next_c = candles[i + 1]

            # Swing High
            if c.high > prev_c.high and c.high > next_c.high:
                swings.append(SwingPoint(index=i, price=c.high, kind="HIGH"))

            # Swing Low
            if c.low < prev_c.low and c.low < next_c.low:
                swings.append(SwingPoint(index=i, price=c.low, kind="LOW"))

        return swings

    def _group_equal_highs(self, swings: List[SwingPoint]) -> List[List[SwingPoint]]:
        """
        Находим группы swing-high'ов с почти одинаковой ценой (EQH).
        """
        highs = [s for s in swings if s.kind == "HIGH"]
        groups: List[List[SwingPoint]] = []
        used = set()

        for i, s in enumerate(highs):
            if i in used:
                continue
            group = [s]
            for j in range(i + 1, len(highs)):
                if j in used:
                    continue
                s2 = highs[j]
                rel_diff = abs(s2.price - s.price) / s.price
                if rel_diff <= self.eq_tolerance:
                    group.append(s2)
                    used.add(j)
            if len(group) >= 2:
                groups.append(group)

        return groups

    def _group_equal_lows(self, swings: List[SwingPoint]) -> List[List[SwingPoint]]:
        """
        Аналогично EQH, но для swing-lows (EQL).
        """
        lows = [s for s in swings if s.kind == "LOW"]
        groups: List[List[SwingPoint]] = []
        used = set()

        for i, s in enumerate(lows):
            if i in used:
                continue
            group = [s]
            for j in range(i + 1, len(lows)):
                if j in used:
                    continue
                s2 = lows[j]
                rel_diff = abs(s2.price - s.price) / s.price
                if rel_diff <= self.eq_tolerance:
                    group.append(s2)
                    used.add(j)
            if len(group) >= 2:
                groups.append(group)

        return groups

    def _detect_bullish_idm(self, candles: List[Candle], swings: List[SwingPoint]) -> Optional[Zone]:
        """
        Bullish IDM (inducement to short → реальный long):
        - Equal lows → ликвидность снизу
        - Свеча делает небольшой прокол ниже этих lows и закрывается выше
        - После свипа появляется бычий displacement
        """
        if len(candles) < 50:
            return None

        avg_b = avg_body(candles, lookback=50)
        if avg_b <= 0:
            return None

        eql_groups = self._group_equal_lows(swings)
        if not eql_groups:
            return None

        n = len(candles)

        for group in reversed(eql_groups):
            base_idx = group[-1].index
            level = min(s.price for s in group)

            max_sweep_idx = min(n - 2, base_idx + self.max_lookahead_sweep)

            for i in range(base_idx + 1, max_sweep_idx + 1):
                c = candles[i]

                if c.low < level:
                    rel_break = (level - c.low) / level
                    if not (self.sweep_min <= rel_break <= self.sweep_max):
                        continue
                    if c.close <= level:
                        continue

                    displacement_found = False
                    local_high = max(candles[j].high for j in range(base_idx, i + 1))

                    max_disp_idx = min(n - 1, i + self.max_lookahead_displacement)
                    for k in range(i + 1, max_disp_idx + 1):
                        ck = candles[k]
                        body = abs(ck.close - ck.open)
                        if ck.close > ck.open and body >= avg_b * self.min_body_factor and ck.close > local_high:
                            displacement_found = True
                            break

                    if not displacement_found:
                        continue

                    zone_low = c.low
                    zone_high = level
                    return Zone(
                        tf="",
                        high=zone_high,
                        low=zone_low,
                        type="IDM_BULL",
                        timestamp=get_candle_timestamp(c),
                        candle_index=i
                    )

        return None

    def _detect_bearish_idm(self, candles: List[Candle], swings: List[SwingPoint]) -> Optional[Zone]:
        """
        Bearish IDM (inducement to long → реальный short):
        - Equal highs → ликвидность сверху
        - Свеча делает небольшой прокол выше этих highs и закрывается ниже
        - После свипа появляется медвежий displacement
        """
        if len(candles) < 50:
            return None

        avg_b = avg_body(candles, lookback=50)
        if avg_b <= 0:
            return None

        eqh_groups = self._group_equal_highs(swings)
        if not eqh_groups:
            return None

        n = len(candles)

        for group in reversed(eqh_groups):
            base_idx = group[-1].index
            level = max(s.price for s in group)

            max_sweep_idx = min(n - 2, base_idx + self.max_lookahead_sweep)

            for i in range(base_idx + 1, max_sweep_idx + 1):
                c = candles[i]

                if c.high > level:
                    rel_break = (c.high - level) / level
                    if not (self.sweep_min <= rel_break <= self.sweep_max):
                        continue
                    if c.close >= level:
                        continue

                    displacement_found = False
                    local_low = min(candles[j].low for j in range(base_idx, i + 1))

                    max_disp_idx = min(n - 1, i + self.max_lookahead_displacement)
                    for k in range(i + 1, max_disp_idx + 1):
                        ck = candles[k]
                        body = abs(ck.close - ck.open)
                        if ck.close < ck.open and body >= avg_b * self.min_body_factor and ck.close < local_low:
                            displacement_found = True
                            break

                    if not displacement_found:
                        continue

                    zone_high = c.high
                    zone_low = level
                    return Zone(
                        tf="",
                        high=zone_high,
                        low=zone_low,
                        type="IDM_BEAR",
                        timestamp=get_candle_timestamp(c),
                        candle_index=i
                    )

        return None

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        """Основной метод детекции IDM зон"""
        zones: List[Zone] = []

        if not candles or len(candles) < 50:
            return DetectionResult(zones=zones, context=None)

        swings = self._find_swings(candles)

        bull_zone = self._detect_bullish_idm(candles, swings)
        if bull_zone:
            bull_zone.tf = tf
            zones.append(bull_zone)

        bear_zone = self._detect_bearish_idm(candles, swings)
        if bear_zone:
            bear_zone.tf = tf
            zones.append(bear_zone)

        return DetectionResult(zones=zones, context=None)
