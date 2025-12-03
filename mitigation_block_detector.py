from typing import List, Optional
from dataclasses import dataclass
from analysis_interfaces import Zone, DetectionResult


def _get_candle_timestamp(candle) -> Optional[int]:
    """Безопасно извлекает timestamp из свечи"""
    ts = getattr(candle, 'time', None)
    if ts is None:
        ts = getattr(candle, 'timestamp', None)

    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1000)

    if isinstance(ts, (int, float)):
        if ts < 1000000000000:
            return int(ts * 1000)
        return int(ts)

    import time
    return int(time.time() * 1000)

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

    # ---------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------

    def detect(self, candles: List, tf: str) -> DetectionResult:
        """ИСПРАВЛЕННЫЙ метод detect для MitigationBlockDetector с timestamps"""
        zones = []

        if not candles or len(candles) < 50:
            return DetectionResult([], None)

        bull = self._detect_bullish_mb(candles)
        if bull:
            bull.tf = tf
            # ДОБАВЛЯЕМ timestamp и candle_index
            # Ищем mitigation candle
            for i in range(len(candles) - 1, 0, -1):
                c = candles[i]
                if c.low >= bull.low and c.high <= bull.high:
                    bull.timestamp = _get_candle_timestamp(c)
                    bull.candle_index = i
                    break
            # Fallback
            if bull.timestamp is None and len(candles) > 0:
                bull.timestamp = _get_candle_timestamp(candles[-1])
                bull.candle_index = len(candles) - 1
            zones.append(bull)

        bear = self._detect_bearish_mb(candles)
        if bear:
            bear.tf = tf
            # ДОБАВЛЯЕМ timestamp и candle_index
            # Ищем mitigation candle
            for i in range(len(candles) - 1, 0, -1):
                c = candles[i]
                if c.low >= bear.low and c.high <= bear.high:
                    bear.timestamp = _get_candle_timestamp(c)
                    bear.candle_index = i
                    break
            # Fallback
            if bear.timestamp is None and len(candles) > 0:
                bear.timestamp = _get_candle_timestamp(candles[-1])
                bear.candle_index = len(candles) - 1
            zones.append(bear)

        return DetectionResult(zones, None)

    # ---------------------------------------------------
    # BULLISH MB
    # ---------------------------------------------------
    def _detect_bullish_mb(self, candles: List) -> Optional[Zone]:
        n = len(candles)
        look = self.lookback_bos

        # Ищем BOS UP
        bos_idx = self._find_bos_up(candles, look)
        if bos_idx is None:
            return None

        # Находим последнюю медвежью свечу перед BOS
        breaker_idx = self._last_bearish_before(candles, bos_idx - look, bos_idx)
        if breaker_idx is None:
            return None

        c = candles[breaker_idx]
        body_low = min(c.open, c.close)
        body_high = max(c.open, c.close)

        # Ищем ретест после BOS
        retest_idx = self._retest_bullish(candles, bos_idx, body_low, body_high)
        if retest_idx is None:
            return None

        return Zone(
            tf="",
            low=body_low,
            high=body_high,
            type="MB_BULL"
        )

    # ---------------------------------------------------
    # BEARISH MB
    # ---------------------------------------------------
    def _detect_bearish_mb(self, candles: List) -> Optional[Zone]:
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
            type="MB_BEAR"
        )

    # ---------------------------------------------------
    # BOS DETECTION
    # ---------------------------------------------------
    def _find_bos_up(self, candles: List, lookback: int) -> Optional[int]:
        n = len(candles)
        for i in range(n - lookback, n - 1):
            # медвежья структура → бычий импульс
            prev_high = max(c.high for c in candles[i - lookback:i])
            if candles[i].close > prev_high * 1.0001:
                return i
        return None

    def _find_bos_down(self, candles: List, lookback: int) -> Optional[int]:
        n = len(candles)
        for i in range(n - lookback, n - 1):
            prev_low = min(c.low for c in candles[i - lookback:i])
            if candles[i].close < prev_low * 0.9999:
                return i
        return None

    # ---------------------------------------------------
    # LAST CANDLES BEFORE BOS
    # ---------------------------------------------------
    def _last_bearish_before(self, candles: List, start: int, end: int) -> Optional[int]:
        for i in range(end - 1, start - 1, -1):
            if candles[i].close < candles[i].open:
                return i
        return None

    def _last_bullish_before(self, candles: List, start: int, end: int) -> Optional[int]:
        for i in range(end - 1, start - 1, -1):
            if candles[i].close > candles[i].open:
                return i
        return None

    # ---------------------------------------------------
    # RETEST CONDITIONS
    # ---------------------------------------------------
    def _retest_bullish(self, candles, bos_idx, low, high):
        """
        Цена должна:
        - прийти в тело свечи
        - НЕ пробить low свечи
        - сформировать подтверждающую свечу
        """
        for i in range(bos_idx + 1, len(candles)):
            c = candles[i]

            # цена входит в тело
            if c.low <= high and c.high >= low:
                # но НЕ пробивает low
                if c.low < low * 0.9995:
                    return None
                return i
        return None

    def _retest_bearish(self, candles, bos_idx, low, high):
        for i in range(bos_idx + 1, len(candles)):
            c = candles[i]

            if c.high >= low and c.low <= high:
                # не должно быть пробития high тела
                if c.high > high * 1.0005:
                    return None
                return i
        return None
