# signal_deduplicator.py
# Умная дедупликация сигналов с учётом зон и времени

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib


@dataclass
class SignalRecord:
    """Запись о сигнале для дедупликации"""
    symbol: str
    direction: str
    zone_low: float
    zone_high: float
    chain_id: str
    tf: str
    created_at: datetime
    expires_at: datetime

    def get_zone_key(self) -> str:
        """Ключ зоны (без учёта цепочки) - группирует близкие зоны"""
        # Округляем до 0.1% для группировки близких зон
        price_precision = max(self.zone_low * 0.001, 0.01)
        rounded_low = round(self.zone_low / price_precision) * price_precision
        rounded_high = round(self.zone_high / price_precision) * price_precision
        return f"{self.symbol}_{self.direction}_{rounded_low:.2f}_{rounded_high:.2f}"


class SignalDeduplicator:
    """
    Умная дедупликация сигналов с учётом:
    - Зоны входа (не точной цены)
    - Времени жизни сигнала (TTL)
    - Перекрытия зон от разных цепочек
    """

    def __init__(self):
        # Основной кэш: zone_key -> SignalRecord
        self.zone_cache: Dict[str, SignalRecord] = {}

        # Кэш точных сигналов для предотвращения повторной отправки
        self.sent_signals: Dict[str, datetime] = {}

        # TTL по таймфреймам
        self.ttl_by_tf = {
            "15m": timedelta(hours=4),  # 4 часа
            "1h": timedelta(hours=12),  # 12 часов
            "4h": timedelta(days=2),  # 2 дня
            "1d": timedelta(days=7),  # 7 дней
        }

        # Минимальное расстояние между зонами для считания их разными (% от цены)
        self.zone_overlap_threshold = 0.005  # 0.5%

        # Статистика
        self.stats = {
            "duplicates_blocked": 0,
            "signals_registered": 0,
            "cleanups_performed": 0
        }

    def _get_signal_hash(self, signal) -> str:
        """Точный хеш сигнала для предотвращения полных дубликатов"""
        data = f"{signal.symbol}_{signal.chain_id}_{signal.direction}_{signal.entry:.6f}_{signal.stop_loss:.6f}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _zones_overlap(
            self,
            zone1_low: float,
            zone1_high: float,
            zone2_low: float,
            zone2_high: float
    ) -> bool:
        """Проверяет, перекрываются ли две зоны с учётом tolerance"""
        # Добавляем tolerance
        mid_price = (zone1_low + zone1_high + zone2_low + zone2_high) / 4
        tolerance = mid_price * self.zone_overlap_threshold

        # Расширяем зоны на tolerance
        z1_low = zone1_low - tolerance
        z1_high = zone1_high + tolerance
        z2_low = zone2_low - tolerance
        z2_high = zone2_high + tolerance

        # Проверяем перекрытие
        return not (z1_high < z2_low or z1_low > z2_high)

    def _normalize_direction(self, direction) -> str:
        """Нормализует направление к LONG/SHORT"""
        dir_str = str(direction).upper().replace("DIRECTION.", "")
        if dir_str in ["LONG", "BUY"]:
            return "LONG"
        elif dir_str in ["SHORT", "SELL"]:
            return "SHORT"
        return dir_str

    def _get_signal_zone_bounds(self, signal) -> Tuple[float, float]:
        """
        Определяет границы зоны сигнала.
        Для LONG: зона между SL (низ) и Entry (верх)
        Для SHORT: зона между Entry (низ) и SL (верх)
        """
        direction = self._normalize_direction(signal.direction)

        if direction == "LONG":
            return (signal.stop_loss, signal.entry)
        else:  # SHORT
            return (signal.entry, signal.stop_loss)

    def is_duplicate(self, signal, tf: str = "15m") -> Tuple[bool, str]:
        """
        Проверяет, является ли сигнал дубликатом.

        Args:
            signal: Объект сигнала с полями symbol, direction, entry, stop_loss, chain_id
            tf: Таймфрейм сигнала

        Returns:
            (is_duplicate: bool, reason: str)
        """
        now = datetime.now()

        # 1. Очищаем истекшие записи (не каждый раз, для производительности)
        if len(self.zone_cache) > 100 or len(self.sent_signals) > 500:
            self._cleanup_expired(now)

        # 2. Проверяем точный хеш (полное совпадение параметров)
        signal_hash = self._get_signal_hash(signal)
        if signal_hash in self.sent_signals:
            self.stats["duplicates_blocked"] += 1
            return True, "Exact duplicate (same entry/SL/chain)"

        # 3. Проверяем перекрытие зон
        direction = self._normalize_direction(signal.direction)
        signal_zone_low, signal_zone_high = self._get_signal_zone_bounds(signal)

        for zone_key, record in self.zone_cache.items():
            # Пропускаем если другой символ
            if record.symbol != signal.symbol:
                continue

            # Пропускаем если другое направление
            if record.direction != direction:
                continue

            # Проверяем истечение записи
            if record.expires_at < now:
                continue

            # Проверяем перекрытие зон
            if self._zones_overlap(
                    signal_zone_low, signal_zone_high,
                    record.zone_low, record.zone_high
            ):
                self.stats["duplicates_blocked"] += 1
                return True, f"Zone overlap with {record.chain_id} ({record.tf}) signal"

        return False, "OK"

    def register_signal(self, signal, tf: str = "15m"):
        """
        Регистрирует отправленный сигнал в кэше.

        Args:
            signal: Объект сигнала
            tf: Таймфрейм сигнала
        """
        now = datetime.now()
        ttl = self.ttl_by_tf.get(tf, timedelta(hours=4))

        direction = self._normalize_direction(signal.direction)
        zone_low, zone_high = self._get_signal_zone_bounds(signal)

        record = SignalRecord(
            symbol=signal.symbol,
            direction=direction,
            zone_low=zone_low,
            zone_high=zone_high,
            chain_id=signal.chain_id,
            tf=tf,
            created_at=now,
            expires_at=now + ttl
        )

        # Добавляем в кэш зон
        zone_key = record.get_zone_key()
        self.zone_cache[zone_key] = record

        # Добавляем точный хеш
        signal_hash = self._get_signal_hash(signal)
        self.sent_signals[signal_hash] = now

        self.stats["signals_registered"] += 1

    def _cleanup_expired(self, now: datetime):
        """Удаляет истекшие записи из кэшей"""
        # Очистка zone_cache
        expired_zones = [k for k, v in self.zone_cache.items() if v.expires_at < now]
        for k in expired_zones:
            del self.zone_cache[k]

        # Очистка sent_signals (храним 24 часа)
        cutoff = now - timedelta(hours=24)
        expired_signals = [k for k, v in self.sent_signals.items() if v < cutoff]
        for k in expired_signals:
            del self.sent_signals[k]

        self.stats["cleanups_performed"] += 1

    def force_cleanup(self):
        """Принудительная очистка кэша"""
        self._cleanup_expired(datetime.now())

    def cleanup(self):
        """Алиас для force_cleanup - для совместимости"""
        self.force_cleanup()

    def clear_symbol(self, symbol: str):
        """Очищает все записи для конкретного символа"""
        keys_to_remove = [k for k, v in self.zone_cache.items() if v.symbol == symbol]
        for k in keys_to_remove:
            del self.zone_cache[k]

    def get_stats(self) -> dict:
        """Возвращает статистику дедупликатора"""
        return {
            "active_zones": len(self.zone_cache),
            "sent_signals_cache": len(self.sent_signals),
            "duplicates_blocked": self.stats["duplicates_blocked"],
            "signals_registered": self.stats["signals_registered"],
            "zones_by_symbol": self._count_by_symbol(),
            "zones_by_direction": self._count_by_direction()
        }

    def _count_by_symbol(self) -> dict:
        """Подсчёт активных зон по символам"""
        counts = {}
        for record in self.zone_cache.values():
            counts[record.symbol] = counts.get(record.symbol, 0) + 1
        return counts

    def _count_by_direction(self) -> dict:
        """Подсчёт активных зон по направлениям"""
        counts = {"LONG": 0, "SHORT": 0}
        for record in self.zone_cache.values():
            counts[record.direction] = counts.get(record.direction, 0) + 1
        return counts

    def get_active_zones(self, symbol: str = None) -> list:
        """Возвращает список активных зон (для отладки)"""
        now = datetime.now()
        zones = []

        for record in self.zone_cache.values():
            if record.expires_at < now:
                continue
            if symbol and record.symbol != symbol:
                continue

            zones.append({
                "symbol": record.symbol,
                "direction": record.direction,
                "zone": f"{record.zone_low:.2f} - {record.zone_high:.2f}",
                "chain": record.chain_id,
                "tf": record.tf,
                "expires_in": str(record.expires_at - now)
            })

        return zones