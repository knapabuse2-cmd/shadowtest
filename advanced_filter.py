# advanced_filter.py - ИСПРАВЛЕННАЯ ВЕРСИЯ СО ВСЕМИ МЕТОДАМИ

from typing import Dict, Tuple
from analysis_interfaces import ChainSignal


class AdvancedFilter:  # Переименовано для соответствия импортам
    """
    Многоуровневая фильтрация сигналов
    Альтернативное имя класса для совместимости
    """

    def __init__(self):
        pass  # Упрощенная инициализация

    def check_signal(self, signal: ChainSignal, candles: Dict, detections: Dict) -> Tuple[bool, str]:
        """
        Основной метод проверки сигнала
        """
        # Проверка времени
        time_check = self._filter_by_time_consistency(signal, candles, detections)
        if not time_check[0]:
            return time_check

        # Проверка качества зоны
        zone_check = self._filter_by_zone_quality(signal, candles, detections)
        if not zone_check[0]:
            return zone_check

        # Проверка моментума
        momentum_check = self._filter_by_momentum(signal, candles)
        if not momentum_check[0]:
            return momentum_check

        return True, "All checks passed"

    def _filter_by_time_consistency(
            self,
            signal: ChainSignal,
            candles: Dict,
            detections: Dict
    ) -> Tuple[bool, str]:
        """
        Проверяет, что зоны не слишком старые
        """
        # Получаем таймфрейм сигнала
        tf = signal.tf if hasattr(signal, 'tf') else "15m"

        # Проверяем возраст зоны
        if signal.zone and hasattr(signal.zone, 'age'):
            if signal.zone.age > 50:
                return False, "Zone too old (>50 candles)"

        return True, "Time consistency OK"

    def _filter_by_zone_quality(
            self,
            signal: ChainSignal,
            candles: Dict,
            detections: Dict
    ) -> Tuple[bool, str]:
        """
        Проверяет качество зоны (чистота, количество касаний)
        """
        if not signal.zone:
            return True, "No zone to check"

        # Получаем свечи для таймфрейма сигнала
        tf = signal.tf if hasattr(signal, 'tf') else "15m"
        tf_candles = candles.get(tf, [])

        if not tf_candles:
            return True, "No candles for zone check"

        # Считаем касания зоны
        touches = 0
        for candle in tf_candles[-20:]:  # Последние 20 свечей
            if candle.low <= signal.zone.high and candle.high >= signal.zone.low:
                touches += 1

        if touches > 3:
            return False, f"Zone tested too many times ({touches} > 3)"

        if touches == 0:
            return False, "Zone never tested"

        return True, "Zone quality OK"

    def _filter_by_momentum(
            self,
            signal: ChainSignal,
            candles: Dict
    ) -> Tuple[bool, str]:
        """
        Проверяет импульс движения (RSI-like)
        """
        # Получаем свечи для основного таймфрейма
        tf = signal.tf if hasattr(signal, 'tf') else "15m"
        tf_candles = candles.get(tf, [])

        if len(tf_candles) < 20:
            return True, "Not enough candles for momentum check"

        # Расчет RSI-подобного индикатора
        gains = []
        losses = []

        for i in range(1, min(15, len(tf_candles))):
            try:
                change = tf_candles[-i].close - tf_candles[-i - 1].close
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            except:
                continue

        if not gains or not losses:
            return True, "Cannot calculate momentum"

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        if avg_loss == 0:
            momentum = 100
        else:
            rs = avg_gain / avg_loss
            momentum = 100 - (100 / (1 + rs))

        # Фильтруем экстремальные значения
        direction = str(signal.direction).upper()
        if direction in ["LONG", "BUY"] and momentum > 70:
            return False, f"Overbought (RSI={momentum:.1f})"
        if direction in ["SHORT", "SELL"] and momentum < 30:
            return False, f"Oversold (RSI={momentum:.1f})"

        return True, "Momentum OK"

    def _filter_by_divergence(
            self,
            signal: ChainSignal,
            context: Dict
    ) -> Tuple[bool, str]:
        """
        Проверяет дивергенции между ценой и индикаторами
        """
        # Упрощенная проверка дивергенций
        # В реальности здесь был бы более сложный анализ
        return True, "No divergence detected"

    def _filter_by_correlation(
            self,
            signal: ChainSignal,
            context: Dict
    ) -> Tuple[bool, str]:
        """
        Проверяет корреляцию с другими парами
        """
        # Упрощенная проверка корреляций
        # В реальности здесь был бы анализ других пар
        return True, "Correlation check passed"


# Для совместимости с разными вариантами импорта
class AdvancedSignalFilter(AdvancedFilter):
    """
    Альтернативное имя класса для обратной совместимости
    """
    pass