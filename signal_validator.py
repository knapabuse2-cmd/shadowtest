# signal_validator.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# ============================================
# УБРАНЫ процентные лимиты на SL!
# SL определяется структурно в analysis_chains.py

from typing import List, Optional, Set, Tuple
from analysis_interfaces import ChainSignal, VolumeContext
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None


class SignalValidator:
    """
    Валидация сигналов БЕЗ процентных лимитов на SL.
    SL определяется структурно (за swing/POI).
    """

    def __init__(self):
        # Минимальный RR
        self.min_rr = 1.5

        # Спред
        self.max_spread_percent = 0.003  # 0.3%

        # Минимальный размер зоны
        self.min_zone_size_percent = 0.0005  # 0.05%

        # УБРАНЫ ЛИМИТЫ НА SL!
        # self.min_sl_distance_percent = НЕТ
        # self.max_sl_distance_percent = НЕТ

        self.signal_cache: Set[str] = set()
        self.active_positions: dict = {}

    def _get_signal_hash(self, signal: ChainSignal) -> str:
        """Создает хеш сигнала для проверки дубликатов"""
        tp_value = signal.take_profits[0] if signal.take_profits else 0.0
        return f"{signal.symbol}_{signal.entry:.5f}_{signal.stop_loss:.5f}_{tp_value:.5f}"

    def _has_conflict(self, signal: ChainSignal, all_signals: List[ChainSignal]) -> Tuple[bool, str]:
        """Проверяет конфликты с другими сигналами"""
        for other in all_signals:
            if other == signal:
                continue

            if other.symbol == signal.symbol:
                if other.direction != signal.direction:
                    entry_diff = abs(other.entry - signal.entry) / signal.entry
                    if entry_diff < 0.01:
                        return True, f"Conflicting directions for {signal.symbol}"

                if abs(other.entry - signal.entry) < 0.00001:
                    if abs(other.stop_loss - signal.stop_loss) < 0.00001:
                        if other.rr >= signal.rr:
                            return True, f"Duplicate signal with worse RR"

        return False, ""

    def validate_signal(self, signal: ChainSignal, context: Optional[VolumeContext] = None) -> ValidationResult:
        """
        Проверка сигнала на валидность.
        БЕЗ процентных лимитов на SL - стоп там где должен быть структурно.
        """

        # 1. Проверка RR
        if signal.rr < self.min_rr:
            return ValidationResult(False, f"RR too low: {signal.rr:.2f} < {self.min_rr}")

        # 2. Проверка размера зоны (минимальный риск)
        zone_size = abs(signal.entry - signal.stop_loss)
        zone_percent = zone_size / signal.entry if signal.entry != 0 else 0
        if zone_percent < self.min_zone_size_percent:
            return ValidationResult(False, f"Zone too small: {zone_percent:.4%}")

        # 3. Bias filtering (только для STRONG bias)
        if context and context.bias != "RANGE":
            if context.bias == "STRONG_BULLISH" and signal.direction in ["SHORT", "SELL"]:
                if hasattr(context, 'structure') and "HH" in context.structure and "HL" in context.structure:
                    return ValidationResult(False, "Strong bullish structure (HH+HL), rejecting SHORT")

            if context.bias == "STRONG_BEARISH" and signal.direction in ["LONG", "BUY"]:
                if hasattr(context, 'structure') and "LH" in context.structure and "LL" in context.structure:
                    return ValidationResult(False, "Strong bearish structure (LH+LL), rejecting LONG")

        # 4. Проверка корректности направления SL/TP
        direction_str = str(signal.direction).upper()
        if "LONG" in direction_str or "BUY" in direction_str:
            if signal.take_profits and any(tp <= signal.entry for tp in signal.take_profits):
                return ValidationResult(False, "Invalid TP for LONG (TP <= Entry)")
            if signal.stop_loss >= signal.entry:
                return ValidationResult(False, "Invalid SL for LONG (SL >= Entry)")
        else:  # SHORT/SELL
            if signal.take_profits and any(tp >= signal.entry for tp in signal.take_profits):
                return ValidationResult(False, "Invalid TP for SHORT (TP >= Entry)")
            if signal.stop_loss <= signal.entry:
                return ValidationResult(False, "Invalid SL for SHORT (SL <= Entry)")

        # 5. БЕЗ ПРОВЕРКИ ПРОЦЕНТОВ SL!
        # Стоп определяется структурно в analysis_chains.py
        # Там он ставится за swing или POI - это правильное место

        # 6. Проверка что TP не слишком близко (минимум 0.05%)
        if signal.take_profits:
            tp_distance = abs(signal.take_profits[0] - signal.entry)
            tp_percent = tp_distance / signal.entry if signal.entry != 0 else 0

            if tp_percent < 0.0005:  # 0.05%
                return ValidationResult(False, f"TP too close: {tp_percent:.2%}")

        # 7. Проверка RR (TP1 минимум 1.5x от SL)
        if signal.take_profits and len(signal.take_profits) > 0:
            tp1_distance = abs(signal.take_profits[0] - signal.entry)
            sl_distance = abs(signal.entry - signal.stop_loss)

            if sl_distance > 0:
                tp_sl_ratio = tp1_distance / sl_distance
                if tp_sl_ratio < 1.5:
                    return ValidationResult(False, f"TP1/SL ratio too low: {tp_sl_ratio:.2f}")

        return ValidationResult(True)

    def filter_signals(self, signals: List[ChainSignal], contexts: dict = None) -> List[ChainSignal]:
        """
        УЛУЧШЕННАЯ фильтрация списка сигналов
        """
        if not signals:
            return []

        # Первый проход - базовая валидация
        pre_validated = []
        for sig in signals:
            ctx = contexts.get(sig.tf) if contexts else None
            result = self.validate_signal(sig, ctx)
            if result.is_valid:
                pre_validated.append(sig)
            else:
                print(f"  ✗ {sig.chain_id} rejected: {result.reason}")

        if not pre_validated:
            return []

        # Второй проход - удаление конфликтов и дубликатов
        final_signals = []
        seen_symbols_directions = {}  # symbol -> direction -> best_signal

        for sig in pre_validated:
            # Нормализуем направление
            direction_str = str(sig.direction).upper().replace("DIRECTION.", "")
            key = (sig.symbol, direction_str)

            # Проверяем, есть ли уже сигнал для этой пары symbol/direction
            if key in seen_symbols_directions:
                existing = seen_symbols_directions[key]

                # Проверяем, не дубликат ли это
                entry_diff = abs(existing.entry - sig.entry) / sig.entry if sig.entry != 0 else 1
                sl_diff = abs(existing.stop_loss - sig.stop_loss) / sig.stop_loss if sig.stop_loss != 0 else 1

                # Более строгая проверка дубликатов
                if entry_diff < 0.0005 and sl_diff < 0.0005:  # Практически идентичные (0.05%)
                    # Оставляем с лучшим RR
                    if sig.rr > existing.rr * 1.1:  # Новый сигнал лучше на 10%+
                        seen_symbols_directions[key] = sig
                        final_signals = [s for s in final_signals if s != existing]
                        final_signals.append(sig)
                        print(
                            f"  ↻ Replacing {existing.chain_id} with {sig.chain_id} (RR: {existing.rr:.2f} → {sig.rr:.2f})")
                else:
                    # Разные зоны входа - можем оставить оба если не конфликтуют
                    if entry_diff > 0.02:  # Больше 2% разницы - разные зоны
                        final_signals.append(sig)
            else:
                # Первый сигнал для этой пары
                seen_symbols_directions[key] = sig
                final_signals.append(sig)

        # Третий проход - проверка на противоположные сигналы
        symbols_with_conflicts = set()
        for i, sig1 in enumerate(final_signals):
            for sig2 in final_signals[i + 1:]:
                if sig1.symbol == sig2.symbol:
                    dir1 = str(sig1.direction).upper().replace("DIRECTION.", "")
                    dir2 = str(sig2.direction).upper().replace("DIRECTION.", "")

                    # Проверяем противоположные направления
                    is_opposite = False
                    if ("LONG" in dir1 or "BUY" in dir1) and ("SHORT" in dir2 or "SELL" in dir2):
                        is_opposite = True
                    elif ("SHORT" in dir1 or "SELL" in dir1) and ("LONG" in dir2 or "BUY" in dir2):
                        is_opposite = True

                    if is_opposite:
                        # Есть конфликт направлений
                        entry_diff = abs(sig1.entry - sig2.entry) / sig1.entry if sig1.entry != 0 else 1

                        # Разрешаем противоположные сигналы если зоны далеко
                        if entry_diff < 0.01:  # Зоны входа близки (< 1%)
                            symbols_with_conflicts.add(sig1.symbol)
                            print(f"  ⚠ Conflict detected for {sig1.symbol}: {dir1} vs {dir2} (entries too close)")
                        else:
                            # Зоны далеко - это могут быть разные уровни, разрешаем оба
                            print(f"  ✓ Allowing both {dir1} and {dir2} for {sig1.symbol} (different zones)")

        # Удаляем все сигналы по символам с близкими конфликтующими зонами
        if symbols_with_conflicts:
            final_signals = [s for s in final_signals if s.symbol not in symbols_with_conflicts]
            print(f"  ✗ Removed signals for conflicting symbols: {symbols_with_conflicts}")

        # Обновляем кеш
        for sig in final_signals:
            sig_hash = self._get_signal_hash(sig)
            self.signal_cache.add(sig_hash)

        # Сортируем по RR (лучшие первыми)
        final_signals.sort(key=lambda x: x.rr, reverse=True)

        best_rr_str = f"{final_signals[0].rr:.2f}" if final_signals else "0"
        print(f"  ✓ Final signals: {len(final_signals)} out of {len(signals)} (best RR: {best_rr_str})")

        return final_signals

    def validate_batch(self, signals: List[ChainSignal], context: Optional[VolumeContext] = None) -> List[ChainSignal]:
        """Валидирует список сигналов (упрощённая версия)"""
        valid_signals = []

        for signal in signals:
            result = self.validate_signal(signal, context)
            if result.is_valid:
                has_conflict, _ = self._has_conflict(signal, valid_signals)
                if not has_conflict:
                    valid_signals.append(signal)
                    self.signal_cache.add(self._get_signal_hash(signal))

        valid_signals.sort(key=lambda s: s.rr, reverse=True)
        return valid_signals

    def is_duplicate(self, signal: ChainSignal) -> bool:
        """Проверяет, является ли сигнал дубликатом"""
        signal_hash = self._get_signal_hash(signal)
        if signal_hash in self.signal_cache:
            return True
        self.signal_cache.add(signal_hash)
        return False

    def clear_cache(self):
        """Очищает кэш сигналов"""
        self.signal_cache.clear()

    def clear_old_cache(self, max_size: int = 1000):
        """Очищает кеш если он слишком большой"""
        if len(self.signal_cache) > max_size:
            # Оставляем только последние 50%
            keep_size = max_size // 2
            cache_list = list(self.signal_cache)
            self.signal_cache = set(cache_list[-keep_size:])
            print(f"  🧹 Cleared signal cache: {len(cache_list)} → {len(self.signal_cache)}")

    def has_active_position(self, symbol: str) -> bool:
        """Проверяет, есть ли активная позиция"""
        return symbol in self.active_positions

    def register_position(self, symbol: str, signal: ChainSignal):
        """Добавляет активную позицию"""
        self.active_positions[symbol] = signal

    def unregister_position(self, symbol: str):
        """Удаляет активную позицию"""
        self.active_positions.pop(symbol, None)