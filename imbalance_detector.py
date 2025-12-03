# imbalance_detector.py
# ИСПРАВЛЕНО: Использует общие утилиты из utils.py

from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
from utils import get_candle_timestamp, avg_body
from data.data_interfaces import Candle


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

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        """Основной метод детекции"""
        zones: List[Zone] = []

        if not candles or len(candles) < 20:
            return DetectionResult([], None)

        avg_b = avg_body(candles, self.lookback_for_avg)
        if avg_b <= 0:
            return DetectionResult([], None)

        bisi_list = self._detect_bisi(candles, avg_b, tf)
        sibi_list = self._detect_sibi(candles, avg_b, tf)

        zones.extend(bisi_list)
        zones.extend(sibi_list)

        return DetectionResult(zones, None)

    def _detect_bisi(self, candles: List[Candle], avg_b: float, tf: str) -> List[Zone]:
        """Buy-side Imbalance (bullish)"""
        zones: List[Zone] = []
        n = len(candles)

        for i in range(1, n - 1):
            prev = candles[i - 1]
            cur = candles[i]
            nxt = candles[i + 1]

            cur_body = abs(cur.close - cur.open)
            nxt_body = abs(nxt.close - nxt.open)

            if max(cur_body, nxt_body) < avg_b * self.min_displacement_factor:
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

            zones.append(
                Zone(
                    tf=tf,
                    low=gap_high,
                    high=gap_low,
                    type="BISI",
                    timestamp=get_candle_timestamp(nxt),
                    candle_index=i + 1
                )
            )

        return zones

    def _detect_sibi(self, candles: List[Candle], avg_b: float, tf: str) -> List[Zone]:
        """Sell-side Imbalance (bearish)"""
        zones: List[Zone] = []
        n = len(candles)

        for i in range(1, n - 1):
            prev = candles[i - 1]
            cur = candles[i]
            nxt = candles[i + 1]

            cur_body = abs(cur.close - cur.open)
            nxt_body = abs(nxt.close - nxt.open)

            if max(cur_body, nxt_body) < avg_b * self.min_displacement_factor:
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

            zones.append(
                Zone(
                    tf=tf,
                    low=gap_low,
                    high=gap_high,
                    type="SIBI",
                    timestamp=get_candle_timestamp(nxt),
                    candle_index=i + 1
                )
            )

        return zones
