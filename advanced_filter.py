# advanced_filter.py
# ИСПРАВЛЕНО: Убраны проверки несуществующих атрибутов

from typing import Dict, Tuple, List
from analysis_interfaces import ChainSignal, Zone


class AdvancedSignalFilter:
    """
    Многоуровневая фильтрация сигналов
    """

    def __init__(self):
        pass

    def filter_by_zone_age(
            self,
            signal: ChainSignal,
            zones: List[Zone],
            candles: List
    ) -> Tuple[bool, str]:
        """
        ИСПРАВЛЕННАЯ проверка возраста зон.
        Проверяет candle_index зоны, не несуществующий атрибут signal.zone
        """
        if not candles:
            return True, "OK"

        current_candle_index = len(candles)

        # Лимиты возраста по таймфреймам
        tf_age_limits = {
            "15m": 200,  # 200 свечей = ~50 часов
            "30m": 150,
            "1h": 120,   # 120 свечей = 5 дней
            "4h": 100,   # 100 свечей = ~16 дней
            "1d": 50,    # 50 свечей = 50 дней
        }

        max_age = tf_age_limits.get(signal.tf, 100)

        # Проверяем возраст зон, переданных как параметр
        for zone in zones:
            if hasattr(zone, 'candle_index') and zone.candle_index is not None:
                age = current_candle_index - zone.candle_index

                # Учитываем таймфрейм зоны
                if hasattr(zone, 'tf') and zone.tf:
                    zone_tf_limit = tf_age_limits.get(zone.tf, 100)
                    effective_limit = max(max_age, zone_tf_limit)

                    # Для старших TF увеличиваем лимит
                    if zone.tf in ["4h", "1d"]:
                        effective_limit = int(effective_limit * 1.5)

                    if age > effective_limit:
                        return False, f"Zone too old ({age} candles > {effective_limit} limit for {zone.tf})"
                else:
                    if age > max_age:
                        return False, f"Zone too old ({age} candles > {max_age} limit)"

        return True, "OK"

    def filter_by_momentum(
            self,
            signal: ChainSignal,
            candles: List
    ) -> Tuple[bool, str]:
        """
        Проверяет импульс движения (простой RSI-like)
        """
        if not candles or len(candles) < 20:
            return True, "OK"

        # RSI calculation
        gains = []
        losses = []

        for i in range(1, min(15, len(candles))):
            if i >= len(candles):
                break
            change = candles[-i].close - candles[-i - 1].close
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))

        if not gains and not losses:
            return True, "OK"

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 1e-10

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Фильтруем экстремальные значения
        direction = str(signal.direction).upper()
        if "LONG" in direction and rsi > 75:
            return False, f"Overbought (RSI={rsi:.0f})"
        if "SHORT" in direction and rsi < 25:
            return False, f"Oversold (RSI={rsi:.0f})"

        return True, "OK"

    def filter_by_volatility(
            self,
            signal: ChainSignal,
            candles: List
    ) -> Tuple[bool, str]:
        """
        Проверяет волатильность (ATR)
        """
        if not candles or len(candles) < 20:
            return True, "OK"

        # Простой ATR
        trs = []
        for i in range(1, min(14, len(candles))):
            h = candles[-i].high
            l = candles[-i].low
            prev_close = candles[-i - 1].close
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)

        if not trs:
            return True, "OK"

        atr = sum(trs) / len(trs)
        current_price = candles[-1].close
        atr_percent = (atr / current_price) * 100

        # Слишком низкая волатильность
        if atr_percent < 0.5:
            return False, f"Low volatility (ATR={atr_percent:.2f}%)"

        # Слишком высокая волатильность
        if atr_percent > 5:
            return False, f"High volatility (ATR={atr_percent:.2f}%)"

        return True, "OK"


# Алиас для совместимости
class AdvancedFilter(AdvancedSignalFilter):
    pass
