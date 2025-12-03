# mitigation_block_detector.py
# ИСПРАВЛЕНО: Использует общие утилиты из utils.py

from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
from utils import get_candle_timestamp
from data.data_interfaces import Candle


class MitigationBlockDetector:
    """
    ICT Mitigation Block Detector (PRO version)

    Логика:
    Bullish MB:
        1) Найти BOS UP
        2) Найти последнюю медвежью свечу перед BOS
        3) После BOS должен быть ретест тела этой свечи
        4) Цена не должна пробить low этой свечи во время ретеста

    Bearish MB:
        то же самое наоборот
    """

    def __init__(
        self,
        lookback_bos: int = 30,
        min_displacement_factor: float = 1.5,
        max_retest_distance: float = 0.002
    ):
        self.lookback_bos = lookback_bos
        self.min_displacement_factor = min_displacement_factor
        self.max_retest_distance = max_retest_distance

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        """Основной метод детекции"""
        zones: List[Zone] = []

        if not candles or len(candles) < 50:
            return DetectionResult([], None)

        bull = self._detect_bullish_mb(candles)
        if bull:
            bull.tf = tf
            zones.append(bull)

        bear = self._detect_bearish_mb(candles)
        if bear:
            bear.tf = tf
            zones.append(bear)

        return DetectionResult(zones, None)

    def _detect_bullish_mb(self, candles: List[Candle]) -> Optional[Zone]:
        n = len(candles)
        look = self.lookback_bos

        bos_idx = self._find_bos_up(candles, look)
        if bos_idx is None:
            return None

        breaker_idx = self._last_bearish_before(candles, bos_idx - look, bos_idx)
        if breaker_idx is None:
            return None

        c = candles[breaker_idx]
        body_low = min(c.open, c.close)
        body_high = max(c.open, c.close)

        retest_idx = self._retest_bullish(candles, bos_idx, body_low, body_high)
        if retest_idx is None:
            return None

        return Zone(
            tf="",
            low=body_low,
            high=body_high,
            type="MB_BULL",
            timestamp=get_candle_timestamp(c),
            candle_index=breaker_idx
        )

    def _detect_bearish_mb(self, candles: List[Candle]) -> Optional[Zone]:
        n = len(candles)
        look = self.lookback_bos

        bos_idx = self._find_bos_down(candles, look)
        if bos_idx is None:
            return None

        breaker_idx = self._last_bullish_before(candles, bos_idx - look, bos_idx)
        if breaker_idx is None:
            return None

        c = candles[breaker_idx]
        body_low = min(c.open, c.close)
        body_high = max(c.open, c.close)

        retest_idx = self._retest_bearish(candles, bos_idx, body_low, body_high)
        if retest_idx is None:
            return None

        return Zone(
            tf="",
            low=body_low,
            high=body_high,
            type="MB_BEAR",
            timestamp=get_candle_timestamp(c),
            candle_index=breaker_idx
        )

    def _find_bos_up(self, candles: List[Candle], lookback: int) -> Optional[int]:
        n = len(candles)
        for i in range(n - lookback, n - 1):
            prev_high = max(c.high for c in candles[i - lookback:i])
            if candles[i].close > prev_high * 1.0001:
                return i
        return None

    def _find_bos_down(self, candles: List[Candle], lookback: int) -> Optional[int]:
        n = len(candles)
        for i in range(n - lookback, n - 1):
            prev_low = min(c.low for c in candles[i - lookback:i])
            if candles[i].close < prev_low * 0.9999:
                return i
        return None

    def _last_bearish_before(self, candles: List[Candle], start: int, end: int) -> Optional[int]:
        for i in range(end - 1, start - 1, -1):
            if candles[i].close < candles[i].open:
                return i
        return None

    def _last_bullish_before(self, candles: List[Candle], start: int, end: int) -> Optional[int]:
        for i in range(end - 1, start - 1, -1):
            if candles[i].close > candles[i].open:
                return i
        return None

    def _retest_bullish(self, candles: List[Candle], bos_idx: int, low: float, high: float) -> Optional[int]:
        """Цена должна прийти в тело свечи, НЕ пробить low, сформировать подтверждение"""
        for i in range(bos_idx + 1, len(candles)):
            c = candles[i]
            if c.low <= high and c.high >= low:
                if c.low < low * 0.9995:
                    return None
                return i
        return None

    def _retest_bearish(self, candles: List[Candle], bos_idx: int, low: float, high: float) -> Optional[int]:
        for i in range(bos_idx + 1, len(candles)):
            c = candles[i]
            if c.high >= low and c.low <= high:
                if c.high > high * 1.0005:
                    return None
                return i
        return None
