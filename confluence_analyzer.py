# confluence_analyzer.py

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ConfluenceScore:
    symbol: str
    zone: Zone
    score: float  # 0-100
    timeframes_aligned: List[str]
    supporting_factors: List[str]


class ConfluenceAnalyzer:
    """
    Анализирует совпадение зон на разных TF
    """

    def analyze_confluence(
            self,
            detections: Dict[str, DetectionResult],
            candles: Dict[str, List[Candle]]
    ) -> List[ConfluenceScore]:

        confluence_zones = []

        # Проверяем каждую зону на младшем TF
        for zone_15m in detections.get("15m", DetectionResult([], None)).zones:
            score = 0
            aligned_tfs = ["15m"]
            factors = []

            # Проверяем перекрытие с зонами старших TF
            for tf in ["1h", "4h", "1d"]:
                if tf not in detections:
                    continue

                for zone_htf in detections[tf].zones:
                    if self._zones_overlap(zone_15m, zone_htf):
                        score += 25  # +25 за каждый TF
                        aligned_tfs.append(tf)
                        factors.append(f"{tf} {zone_htf.type}")

                        # Bonus за одинаковый тип зоны
                        if zone_15m.type == zone_htf.type:
                            score += 10
                            factors.append(f"Type match on {tf}")

            # Проверяем расположение относительно ключевых уровней
            current_price = candles["15m"][-1].close

            # Зона у round number
            if self._near_round_number(zone_15m):
                score += 15
                factors.append("Round number")

            # Зона у дневного high/low
            daily_high = max(c.high for c in candles["1d"][-1:])
            daily_low = min(c.low for c in candles["1d"][-1:])

            if abs(zone_15m.high - daily_high) / daily_high < 0.001:
                score += 20
                factors.append("Daily high")
            elif abs(zone_15m.low - daily_low) / daily_low < 0.001:
                score += 20
                factors.append("Daily low")

            if score >= 50:  # Минимальный порог
                confluence_zones.append(
                    ConfluenceScore(
                        symbol="",
                        zone=zone_15m,
                        score=min(100, score),
                        timeframes_aligned=aligned_tfs,
                        supporting_factors=factors
                    )
                )

        # Сортируем по score
        return sorted(confluence_zones, key=lambda x: x.score, reverse=True)

    def _zones_overlap(self, z1: Zone, z2: Zone, tolerance: float = 0.001) -> bool:
        """Проверка пересечения зон с допуском"""
        z1_expanded_high = z1.high * (1 + tolerance)
        z1_expanded_low = z1.low * (1 - tolerance)

        return not (z1_expanded_high < z2.low or z1_expanded_low > z2.high)

    def _near_round_number(self, zone: Zone) -> bool:
        """Проверка близости к круглым числам"""
        for price in [zone.high, zone.low]:
            # Для крипты круглые числа - это 100, 500, 1000 и т.д.
            if price > 1000:
                if price % 1000 < 50 or price % 1000 > 950:
                    return True
            elif price > 100:
                if price % 100 < 5 or price % 100 > 95:
                    return True
            elif price > 10:
                if price % 10 < 0.5 or price % 10 > 9.5:
                    return True
        return False