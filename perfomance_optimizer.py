# performance_optimizer.py

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ChainPerformance:
    chain_id: str
    total_signals: int
    win_rate: float
    avg_rr_achieved: float
    best_timeframe: str
    best_session: str  # Время суток
    worst_conditions: List[str]
    optimal_parameters: Dict


class PerformanceOptimizer:
    """
    Отслеживает эффективность каждой цепочки и оптимизирует параметры
    """

    def __init__(self, data_file: str = "performance_data.json"):
        self.data_file = data_file
        self.performance_data = self._load_data()
        self.min_samples = 30  # Минимум сделок для статистики

    def _load_data(self) -> Dict:
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.performance_data, f, indent=2)

    def update_signal_result(
            self,
            chain_id: str,
            symbol: str,
            tf: str,
            outcome: str,  # "TP", "SL", "BE"
            rr_achieved: float,
            entry_time: datetime,
            market_conditions: Dict  # bias, volatility, etc
    ):
        """
        Обновляет статистику после закрытия позиции
        """
        key = f"{chain_id}_{symbol}"

        if key not in self.performance_data:
            self.performance_data[key] = {
                "signals": [],
                "stats": {}
            }

        # Сохраняем результат
        self.performance_data[key]["signals"].append({
            "timestamp": entry_time.isoformat(),
            "tf": tf,
            "outcome": outcome,
            "rr_achieved": rr_achieved,
            "hour": entry_time.hour,
            "day_of_week": entry_time.weekday(),
            "conditions": market_conditions
        })

        # Пересчитываем статистику
        self._recalculate_stats(key)
        self._save_data()

    def _recalculate_stats(self, key: str):
        """
        Пересчитывает статистику для цепочки
        """
        signals = self.performance_data[key]["signals"]

        if len(signals) < self.min_samples:
            return

        # Win rate
        wins = sum(1 for s in signals if s["outcome"] == "TP")
        win_rate = wins / len(signals)

        # Средний RR
        avg_rr = sum(s["rr_achieved"] for s in signals) / len(signals)

        # Лучший таймфрейм
        tf_performance = {}
        for s in signals:
            tf = s["tf"]
            if tf not in tf_performance:
                tf_performance[tf] = {"wins": 0, "total": 0}
            tf_performance[tf]["total"] += 1
            if s["outcome"] == "TP":
                tf_performance[tf]["wins"] += 1

        best_tf = max(
            tf_performance.items(),
            key=lambda x: x[1]["wins"] / x[1]["total"] if x[1]["total"] > 10 else 0
        )[0]

        # Лучшее время суток
        hour_performance = {}
        for s in signals:
            hour = s["hour"]
            if hour not in hour_performance:
                hour_performance[hour] = {"wins": 0, "total": 0}
            hour_performance[hour]["total"] += 1
            if s["outcome"] == "TP":
                hour_performance[hour]["wins"] += 1

        best_hours = sorted(
            hour_performance.items(),
            key=lambda x: x[1]["wins"] / x[1]["total"] if x[1]["total"] > 5 else 0,
            reverse=True
        )[:3]  # Top 3 часа

        self.performance_data[key]["stats"] = {
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "best_tf": best_tf,
            "best_hours": [h[0] for h in best_hours],
            "total_signals": len(signals)
        }

    def should_take_signal(
            self,
            chain_id: str,
            symbol: str,
            tf: str,
            current_time: datetime
    ) -> Tuple[bool, str]:
        """
        Решает, стоит ли брать сигнал на основе исторической эффективности
        """
        key = f"{chain_id}_{symbol}"

        if key not in self.performance_data:
            return True, "No history"

        stats = self.performance_data[key].get("stats", {})

        if not stats or stats.get("total_signals", 0) < self.min_samples:
            return True, "Insufficient data"

        # Проверяем win rate
        if stats["win_rate"] < 0.35:  # Меньше 35% - отключаем
            return False, f"Low win rate: {stats['win_rate']:.1%}"

        # Проверяем, хороший ли час для торговли
        current_hour = current_time.hour
        best_hours = stats.get("best_hours", [])

        if best_hours and current_hour not in best_hours:
            # Не лучшее время, но не блокируем полностью
            pass

        # Проверяем таймфрейм
        if tf != stats.get("best_tf"):
            # Можно добавить предупреждение
            pass

        return True, "OK"

    def get_chain_ranking(self) -> List[Tuple[str, float]]:
        """
        Возвращает рейтинг цепочек по эффективности
        """
        rankings = []

        for key, data in self.performance_data.items():
            stats = data.get("stats", {})
            if stats and stats.get("total_signals", 0) >= self.min_samples:
                # Формула эффективности: win_rate * avg_rr
                score = stats["win_rate"] * stats.get("avg_rr", 0)
                chain_id = key.split("_")[0]
                rankings.append((chain_id, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)