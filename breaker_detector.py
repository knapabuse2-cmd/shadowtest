# breaker_detector.py
# ИСПРАВЛЕНО: Использует общие утилиты из utils.py

from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
from utils import get_candle_timestamp, avg_body
from data.data_interfaces import Candle


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # "HIGH" или "LOW"


class BreakerDetector:
    """
    PRO-версия ICT Breaker Block Detector

    Логика (упрощённо, но по ICT):

    Bullish Breaker:
      1) Есть два свинга LOW: старый low1 и более свежий low2, причём low2 < low1 (sweep)
      2) После low2 формируется BOS UP (закрытие выше локальной структуры)
      3) Последняя медвежья свеча перед BOS UP = breaker candle
      4) Зона breaker = диапазон тела (можно учитывать и wick)

    Bearish Breaker:
      1) Два свинга HIGH: high1 и более свежий high2, причём high2 > high1 (sweep)
      2) После high2 BOS DOWN (закрытие ниже локальной структуры)
      3) Последняя бычья свеча перед BOS DOWN = breaker candle
      4) Зона breaker = диапазон тела
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        min_sweep_ratio: float = 0.0005,      # минимальный относительный "вынос"
        min_displacement_factor: float = 1.5  # фактор размера тела свечи для BOS
    ):
        self.swing_lookback = swing_lookback
        self.min_sweep_ratio = min_sweep_ratio
        self.min_displacement_factor = min_displacement_factor

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        """Основной метод детекции"""
        zones: List[Zone] = []

        if not candles or len(candles) < self.swing_lookback * 4:
            return DetectionResult([], None)

        swings = self._find_swings(candles)

        bull_zone = self._detect_bullish_breaker(candles, swings)
        if bull_zone:
            bull_zone.tf = tf
            zones.append(bull_zone)

        bear_zone = self._detect_bearish_breaker(candles, swings)
        if bear_zone:
            bear_zone.tf = tf
            zones.append(bear_zone)

        return DetectionResult(zones, None)

    def _find_swings(self, candles: List[Candle]) -> List[SwingPoint]:
        """Находим локальные HIGH/LOW по окну swing_lookback"""
        swings: List[SwingPoint] = []
        n = len(candles)
        L = self.swing_lookback

        if n < 2 * L + 1:
            return swings

        for i in range(L, n - L):
            h = candles[i].high
            l = candles[i].low

            is_high = all(h >= candles[j].high for j in range(i - L, i + L + 1))
            is_low = all(l <= candles[j].low for j in range(i - L, i + L + 1))

            if is_high:
                swings.append(SwingPoint(i, h, "HIGH"))
            if is_low:
                swings.append(SwingPoint(i, l, "LOW"))

        return swings

    def _detect_bullish_breaker(
        self,
        candles: List[Candle],
        swings: List[SwingPoint]
    ) -> Optional[Zone]:
        """
        Ищем pattern:
          LOW (low1) → LOW (swept low2 < low1) → BOS UP → последняя медвежья свеча перед BOS
        """
        lows = [s for s in swings if s.kind == "LOW"]
        if len(lows) < 2:
            return None

        for i in range(len(lows) - 1, 0, -1):
            low2 = lows[i]
            low1 = lows[i - 1]

            if low2.index <= low1.index:
                continue

            if not self._is_sweep(low1.price, low2.price, direction="DOWN"):
                continue

            bos_index = self._find_bos_up(candles, low1.index, low2.index)
            if bos_index is None:
                continue

            breaker_idx = self._last_bearish_before(candles, low2.index, bos_index)
            if breaker_idx is None:
                continue

            c = candles[breaker_idx]

            if not self._has_bullish_displacement(candles, bos_index):
                continue

            body_low = min(c.open, c.close)
            body_high = max(c.open, c.close)

            return Zone(
                tf="",
                low=body_low,
                high=body_high,
                type="BREAKER_BULL",
                timestamp=get_candle_timestamp(c),
                candle_index=breaker_idx
            )

        return None

    def _detect_bearish_breaker(
        self,
        candles: List[Candle],
        swings: List[SwingPoint]
    ) -> Optional[Zone]:
        """
        Ищем pattern:
          HIGH (high1) → HIGH (swept high2 > high1) → BOS DOWN → последняя бычья свеча перед BOS
        """
        highs = [s for s in swings if s.kind == "HIGH"]
        if len(highs) < 2:
            return None

        for i in range(len(highs) - 1, 0, -1):
            high2 = highs[i]
            high1 = highs[i - 1]

            if high2.index <= high1.index:
                continue

            if not self._is_sweep(high1.price, high2.price, direction="UP"):
                continue

            bos_index = self._find_bos_down(candles, high1.index, high2.index)
            if bos_index is None:
                continue

            breaker_idx = self._last_bullish_before(candles, high2.index, bos_index)
            if breaker_idx is None:
                continue

            c = candles[breaker_idx]

            if not self._has_bearish_displacement(candles, bos_index):
                continue

            body_low = min(c.open, c.close)
            body_high = max(c.open, c.close)

            return Zone(
                tf="",
                low=body_low,
                high=body_high,
                type="BREAKER_BEAR",
                timestamp=get_candle_timestamp(c),
                candle_index=breaker_idx
            )

        return None

    def _is_sweep(self, old_price: float, new_price: float, direction: str) -> bool:
        """Проверка sweep ликвидности"""
        if direction == "DOWN":
            return new_price < old_price * (1 - self.min_sweep_ratio)
        else:
            return new_price > old_price * (1 + self.min_sweep_ratio)

    def _find_bos_up(self, candles: List[Candle], idx1: int, idx2: int) -> Optional[int]:
        """Ищем BOS UP после low2"""
        if idx2 >= len(candles) - 2:
            return None

        local_high = max(c.high for c in candles[idx1:idx2 + 1])
        threshold = local_high * 1.0001

        for i in range(idx2 + 1, len(candles)):
            if candles[i].close > threshold:
                return i
        return None

    def _find_bos_down(self, candles: List[Candle], idx1: int, idx2: int) -> Optional[int]:
        """BOS DOWN"""
        if idx2 >= len(candles) - 2:
            return None

        local_low = min(c.low for c in candles[idx1:idx2 + 1])
        threshold = local_low * 0.9999

        for i in range(idx2 + 1, len(candles)):
            if candles[i].close < threshold:
                return i
        return None

    def _last_bearish_before(self, candles: List[Candle], start_idx: int, end_idx: int) -> Optional[int]:
        """Последняя медвежья свеча перед BOS UP"""
        for i in range(end_idx - 1, start_idx - 1, -1):
            if candles[i].close < candles[i].open:
                return i
        return None

    def _last_bullish_before(self, candles: List[Candle], start_idx: int, end_idx: int) -> Optional[int]:
        """Последняя бычья свеча перед BOS DOWN"""
        for i in range(end_idx - 1, start_idx - 1, -1):
            if candles[i].close > candles[i].open:
                return i
        return None

    def _has_bullish_displacement(self, candles: List[Candle], idx: int) -> bool:
        """Проверка displacement для BOS UP"""
        c = candles[idx]
        if c.close <= c.open:
            return False
        avg_b = avg_body(candles)
        if avg_b <= 0:
            return False
        return (c.close - c.open) > avg_b * self.min_displacement_factor

    def _has_bearish_displacement(self, candles: List[Candle], idx: int) -> bool:
        """Проверка displacement для BOS DOWN"""
        c = candles[idx]
        if c.close >= c.open:
            return False
        avg_b = avg_body(candles)
        if avg_b <= 0:
            return False
        return (c.open - c.close) > avg_b * self.min_displacement_factor
