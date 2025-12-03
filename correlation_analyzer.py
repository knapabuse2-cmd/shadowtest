# correlation_analyzer.py

class CorrelationAnalyzer:
    """
    Отслеживает корреляцию между символами
    """

    def __init__(self):
        self.correlation_matrix = {}
        self.update_interval = 3600  # Обновляем каждый час

    def calculate_correlations(
            self,
            symbols: List[str],
            candles_data: Dict[str, List[Candle]]
    ):
        """
        Рассчитывает матрицу корреляций
        """
        import numpy as np

        for sym1 in symbols:
            for sym2 in symbols:
                if sym1 == sym2:
                    continue

                key = f"{sym1}_{sym2}"

                # Берем close цены
                closes1 = [c.close for c in candles_data.get(sym1, [])]
                closes2 = [c.close for c in candles_data.get(sym2, [])]

                if len(closes1) < 100 or len(closes2) < 100:
                    continue

                # Выравниваем длину
                min_len = min(len(closes1), len(closes2))
                closes1 = closes1[-min_len:]
                closes2 = closes2[-min_len:]

                # Считаем корреляцию
                corr = np.corrcoef(closes1, closes2)[0, 1]
                self.correlation_matrix[key] = corr

    def check_correlation_conflict(
            self,
            new_signal: ChainSignal,
            active_signals: List[ChainSignal]
    ) -> bool:
        """
        Проверяет конфликт с активными сигналами на коррелирующих парах
        """
        for active in active_signals:
            key = f"{new_signal.symbol}_{active.symbol}"
            corr = self.correlation_matrix.get(key, 0)

            # Высокая положительная корреляция
            if corr > 0.7:
                # Сигналы должны быть в одном направлении
                if new_signal.direction != active.direction:
                    return True  # Конфликт

            # Высокая отрицательная корреляция
            elif corr < -0.7:
                # Сигналы должны быть в противоположных направлениях
                if new_signal.direction == active.direction:
                    return True  # Конфликт

        return False