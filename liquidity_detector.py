# liquidity_detector.py
# ИСПРАВЛЕНО: Использует общие утилиты из utils.py

from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
from utils import get_candle_timestamp
from data.data_interfaces import Candle


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # "HIGH" или "LOW"


class LiquidityDetector:
    """
    Liquidity Detector PRO v1

    Что делает:
    - Ищет Equal Highs (EQH) и Equal Lows (EQL) по свингам
    - Строит зоны ликвидности (clusters)
    - Ищет sweeps этих зон (SWEEP_HIGH / SWEEP_LOW)
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        eq_tolerance: float = 0.0007,  # ~0.07% цены
        min_cluster_size: int = 2,
        sweep_lookback: int = 80,
        min_sweep_ratio: float = 0.0007  # ~0.07% прокола
    ):
        self.swing_lookback = swing_lookback
        self.eq_tolerance = eq_tolerance
        self.min_cluster_size = min_cluster_size
        self.sweep_lookback = sweep_lookback
        self.min_sweep_ratio = min_sweep_ratio

    def detect(self, candles: List[Candle], tf: str) -> DetectionResult:
        """Основной метод детекции"""
        zones: List[Zone] = []

        if not candles or len(candles) < self.swing_lookback * 4:
            return DetectionResult([], None)

        swings = self._find_swings(candles)

        # EQH / EQL
        eqh_zones = self._detect_eqh(candles, swings, tf)
        eql_zones = self._detect_eql(candles, swings, tf)

        zones.extend(eqh_zones)
        zones.extend(eql_zones)

        # Sweeps этих пулов
        sweep_high_zones = self._detect_sweeps_high(candles, swings, eqh_zones, tf)
        sweep_low_zones = self._detect_sweeps_low(candles, swings, eql_zones, tf)

        zones.extend(sweep_high_zones)
        zones.extend(sweep_low_zones)

        return DetectionResult(zones, None)

    def _find_swings(self, candles: List[Candle]) -> List[SwingPoint]:
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

    def _detect_eqh(self, candles: List[Candle], swings: List[SwingPoint], tf: str) -> List[Zone]:
        """Equal Highs detection"""
        highs = [s for s in swings if s.kind == "HIGH"]
        zones: List[Zone] = []
        n = len(highs)
        if n < self.min_cluster_size:
            return zones

        used = set()

        for i in range(n):
            if i in used:
                continue
            base = highs[i]
            cluster = [base]
            for j in range(i + 1, n):
                if j in used:
                    continue
                other = highs[j]
                mid_price = (base.price + other.price) / 2.0
                if mid_price <= 0:
                    continue
                if abs(base.price - other.price) / mid_price <= self.eq_tolerance:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) >= self.min_cluster_size:
                prices = [p.price for p in cluster]
                low = min(prices)
                high = max(prices)

                last_swing = max(cluster, key=lambda s: s.index)
                timestamp = get_candle_timestamp(candles[last_swing.index]) if last_swing.index < len(candles) else get_candle_timestamp(candles[-1])

                zones.append(
                    Zone(
                        tf=tf,
                        low=low,
                        high=high,
                        type="EQH",
                        timestamp=timestamp,
                        candle_index=last_swing.index
                    )
                )

        return zones

    def _detect_eql(self, candles: List[Candle], swings: List[SwingPoint], tf: str) -> List[Zone]:
        """Equal Lows detection"""
        lows = [s for s in swings if s.kind == "LOW"]
        zones: List[Zone] = []
        n = len(lows)
        if n < self.min_cluster_size:
            return zones

        used = set()

        for i in range(n):
            if i in used:
                continue
            base = lows[i]
            cluster = [base]
            for j in range(i + 1, n):
                if j in used:
                    continue
                other = lows[j]
                mid_price = (base.price + other.price) / 2.0
                if mid_price <= 0:
                    continue
                if abs(base.price - other.price) / mid_price <= self.eq_tolerance:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) >= self.min_cluster_size:
                prices = [p.price for p in cluster]
                low = min(prices)
                high = max(prices)

                last_swing = max(cluster, key=lambda s: s.index)
                timestamp = get_candle_timestamp(candles[last_swing.index]) if last_swing.index < len(candles) else get_candle_timestamp(candles[-1])

                zones.append(
                    Zone(
                        tf=tf,
                        low=low,
                        high=high,
                        type="EQL",
                        timestamp=timestamp,
                        candle_index=last_swing.index
                    )
                )

        return zones

    def _detect_sweeps_high(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        eqh_zones: List[Zone],
        tf: str
    ) -> List[Zone]:
        """Sweep над EQH"""
        zones: List[Zone] = []
        if not eqh_zones:
            return zones

        n = len(candles)
        if n < 5:
            return zones

        for z in eqh_zones:
            eqh_high = z.high
            threshold = eqh_high * (1.0 + self.min_sweep_ratio)

            start = max(0, n - self.sweep_lookback)
            for i in range(start, n):
                c = candles[i]
                if c.high > threshold and c.close < eqh_high:
                    zones.append(
                        Zone(
                            tf=tf,
                            low=eqh_high,
                            high=c.high,
                            type="SWEEP_HIGH",
                            timestamp=get_candle_timestamp(c),
                            candle_index=i
                        )
                    )
                    break

        return zones

    def _detect_sweeps_low(
        self,
        candles: List[Candle],
        swings: List[SwingPoint],
        eql_zones: List[Zone],
        tf: str
    ) -> List[Zone]:
        """Sweep под EQL"""
        zones: List[Zone] = []
        if not eql_zones:
            return zones

        n = len(candles)
        if n < 5:
            return zones

        for z in eql_zones:
            eql_low = z.low
            threshold = eql_low * (1.0 - self.min_sweep_ratio)

            start = max(0, n - self.sweep_lookback)
            for i in range(start, n):
                c = candles[i]
                if c.low < threshold and c.close > eql_low:
                    zones.append(
                        Zone(
                            tf=tf,
                            low=c.low,
                            high=eql_low,
                            type="SWEEP_LOW",
                            timestamp=get_candle_timestamp(c),
                            candle_index=i
                        )
                    )
                    break

        return zones
