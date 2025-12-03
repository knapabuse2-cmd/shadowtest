from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
# Candle тип уже есть в проекте (data_interfaces.Candle),
# здесь важны только поля: open, high, low, close.

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

class ImbalanceDetector:
    """
    ICT BISI / SIBI Detector (PRO-версия)

    Ищем мощные дисбалансы (Buy-side Imbalance / Sell-side Imbalance)
    по 3-свечному паттерну, но с более жёсткими условиями, чем у FVG:

    BISI (bullish):
        - три свечи: prev, cur, next
        - cur и/или next = бычьи displacement-свечи
        - high(prev) < min(low(cur), low(next))  → пустота внизу
        - gap_size / price >= min_gap_ratio

    SIBI (bearish):
        - high/low зеркально
    """

    def __init__(
        self,
        min_gap_ratio: float = 0.0015,      # минимум 0.15% цены
        min_displacement_factor: float = 1.8,  # тело displacement-свечи > 1.8 * avg_body
        lookback_for_avg: int = 50
    ):
        self.min_gap_ratio = min_gap_ratio
        self.min_displacement_factor = min_displacement_factor
        self.lookback_for_avg = lookback_for_avg

    # ---------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------

    def detect(self, candles: List, tf: str) -> DetectionResult:
        """ИСПРАВЛЕННЫЙ метод detect для ImbalanceDetector с timestamps"""
        zones: List[Zone] = []

        if not candles or len(candles) < 20:
            return DetectionResult([], None)

        avg_body = self._avg_body(candles, self.lookback_for_avg)
        if avg_body <= 0:
            return DetectionResult([], None)

        bisi_list = self._detect_bisi(candles, avg_body)
        sibi_list = self._detect_sibi(candles, avg_body)

        for z in bisi_list + sibi_list:
            z.tf = tf
            zones.append(z)

        return DetectionResult(zones, None)

    # ---------------------------------------------------
    # BISI (Buy-side Imbalance, bullish)
    # ---------------------------------------------------
    def _detect_bisi(self, candles: List, avg_body: float) -> List[Zone]:
        """ИСПРАВЛЕННЫЙ _detect_bisi с timestamps"""
        zones: List[Zone] = []
        n = len(candles)

        for i in range(1, n - 1):
            prev = candles[i - 1]
            cur = candles[i]
            nxt = candles[i + 1]

            cur_body = abs(cur.close - cur.open)
            nxt_body = abs(nxt.close - nxt.open)

            if max(cur_body, nxt_body) < avg_body * self.min_displacement_factor:
                continue

            if not (cur.close > cur.open or nxt.close > nxt.open):
                continue

            gap_low = min(cur.low, nxt.low)
            gap_high = prev.high

            if gap_low <= gap_high:
                continue

            gap_size = gap_low - gap_high
            mid_price = (gap_low + gap_high) / 2.0
            if mid_price <= 0:
                continue

            if gap_size / mid_price < self.min_gap_ratio:
                continue

            # ДОБАВЛЯЕМ timestamp и candle_index
            timestamp = _get_candle_timestamp(nxt)

            zones.append(
                Zone(
                    tf="",
                    low=gap_high,
                    high=gap_low,
                    type="BISI",
                    timestamp=timestamp,  # ДОБАВЛЕНО
                    candle_index=i + 1  # ДОБАВЛЕНО (индекс nxt)
                )
            )

        return zones

    # Обновленный метод _detect_sibi:
    def _detect_sibi(self, candles: List, avg_body: float) -> List[Zone]:
        """ИСПРАВЛЕННЫЙ _detect_sibi с timestamps"""
        zones: List[Zone] = []
        n = len(candles)

        for i in range(1, n - 1):
            prev = candles[i - 1]
            cur = candles[i]
            nxt = candles[i + 1]

            cur_body = abs(cur.close - cur.open)
            nxt_body = abs(nxt.close - nxt.open)

            if max(cur_body, nxt_body) < avg_body * self.min_displacement_factor:
                continue

            if not (cur.close < cur.open or nxt.close < nxt.open):
                continue

            gap_high = max(cur.high, nxt.high)
            gap_low = prev.low

            if gap_low >= gap_high:
                continue

            gap_size = gap_high - gap_low
            mid_price = (gap_low + gap_high) / 2.0
            if mid_price <= 0:
                continue

            if gap_size / mid_price < self.min_gap_ratio:
                continue

            # ДОБАВЛЯЕМ timestamp и candle_index
            timestamp = _get_candle_timestamp(nxt)

            zones.append(
                Zone(
                    tf="",
                    low=gap_low,
                    high=gap_high,
                    type="SIBI",
                    timestamp=timestamp,  # ДОБАВЛЕНО
                    candle_index=i + 1  # ДОБАВЛЕНО
                )
            )

        return zones

    # ---------------------------------------------------
    # HELPERS
    # ---------------------------------------------------
    def _avg_body(self, candles: List, lookback: int) -> float:
        if not candles:
            return 0.0
        tail = candles[-lookback:]
        bodies = [abs(c.close - c.open) for c in tail]
        if not bodies:
            return 0.0
        return sum(bodies) / len(bodies)
