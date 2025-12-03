import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum
import time

from data.data_interfaces import Candle
from analysis_interfaces import ChainSignal


class PositionStatus(str, Enum):
    PENDING = "PENDING"  # лимитка, ждём входа
    OPEN = "OPEN"  # позиция открыта
    CANCELLED = "CANCELLED"  # отменена (ушла без нас)
    CLOSED = "CLOSED"  # закрыта по TP или SL


@dataclass
class TrackedPosition:
    """
    Внутреннее представление сигнала, который мы отслеживаем.
    """
    symbol: str
    chain_id: str
    direction: str  # "LONG" / "SHORT"
    entry: float
    stop_loss: float
    take_profits: List[float]
    rr: float
    created_at: int  # ms
    status: PositionStatus = PositionStatus.PENDING
    opened_at: Optional[int] = None
    closed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    outcome: Optional[str] = None  # "TP" / "SL" / "MISSED"
    hit_tp_index: Optional[int] = None
    last_price: Optional[float] = None
    last_update: Optional[int] = None  # ДОБАВЛЕНО: время последнего обновления
    candles_since_signal: int = 0  # ДОБАВЛЕНО: счетчик свечей с момента сигнала
    entry_touched: bool = False  # ДОБАВЛЕНО: флаг касания entry


class PositionTracker:
    """
    Отслеживает, что произошло с сигналами:
    - лимитка (pending)
    - вход (fill)
    - тейки/стоп
    - отмена, если цена ушла в TP1 без входа
    """

    def __init__(self, publisher):
        self.publisher = publisher
        self.positions: Dict[str, List[TrackedPosition]] = {}
        self.debug_mode = False
        self.min_candles_before_check = 2  # ДОБАВЛЕНО: минимум свечей перед проверкой TP/SL

    def register_signal(self, signal: ChainSignal, now_time: int) -> None:
        """
        Вызывается каждый раз, когда бот нашёл новый сигнал и отправил его в канал.
        """
        # Нормализуем direction
        direction = str(signal.direction).upper().replace("DIRECTION.", "")
        if direction not in ["LONG", "SHORT", "BUY", "SELL"]:
            print(f"⚠ Unknown direction: {direction}, using as is")

        pos = TrackedPosition(
            symbol=signal.symbol,
            chain_id=signal.chain_id,
            direction=direction,
            entry=float(signal.entry),
            stop_loss=float(signal.stop_loss),
            take_profits=[float(tp) for tp in signal.take_profits],
            rr=float(signal.rr),
            created_at=now_time,
            last_update=now_time,  # ДОБАВЛЕНО
        )

        self.positions.setdefault(pos.symbol, []).append(pos)

        if self.debug_mode:
            print(
                f"[TRACKER] Registered {pos.symbol} {pos.direction}: Entry={pos.entry:.5f}, SL={pos.stop_loss:.5f}, TPs={pos.take_profits}")

    async def update_with_candle(self, symbol: str, candle: Candle) -> None:
        """
        ИСПРАВЛЕННАЯ ВЕРСИЯ: Правильно отслеживает новые свечи
        """
        if symbol not in self.positions:
            return

        # Поддержка обеих моделей свечей
        ts = getattr(candle, "time", None)
        if ts is None:
            ts = getattr(candle, "timestamp", None)
            if hasattr(ts, "timestamp"):
                ts = int(ts.timestamp() * 1000)
            elif ts:
                ts = int(ts)
            else:
                ts = int(time.time() * 1000)

        high = float(candle.high)
        low = float(candle.low)
        close = float(candle.close)

        if self.debug_mode:
            print(f"\n[TRACKER] {symbol} candle: H={high:.5f}, L={low:.5f}, C={close:.5f}")

        for pos in self.positions.get(symbol, []):
            # Уже закрытые/отменённые не трогаем
            if pos.status in (PositionStatus.CLOSED, PositionStatus.CANCELLED):
                continue

            # ИСПРАВЛЕНО: Правильно увеличиваем счетчик свечей
            if pos.last_update is None:
                # Первое обновление
                pos.last_update = ts
                pos.candles_since_signal = 0
            else:
                # Проверяем, что прошло достаточно времени для новой свечи
                time_diff = abs(ts - pos.last_update)

                # Определяем минимальный интервал свечи в миллисекундах
                # 15m = 900000ms, 1h = 3600000ms, 4h = 14400000ms, 1d = 86400000ms
                min_candle_interval = 60000  # 1 минута минимум

                if time_diff > min_candle_interval:
                    pos.candles_since_signal += 1
                    pos.last_update = ts

                    if self.debug_mode:
                        print(f"[TRACKER] {pos.symbol}: Candle #{pos.candles_since_signal} since signal")

            pos.last_price = close

            if pos.status == PositionStatus.PENDING:
                await self._process_pending(pos, high, low, close, ts)
            elif pos.status == PositionStatus.OPEN:
                await self._process_open(pos, high, low, close, ts)

    async def _process_pending(
            self,
            pos: TrackedPosition,
            high: float,
            low: float,
            close: float,
            ts: Optional[int],
    ) -> None:
        """
        ИСПРАВЛЕННАЯ логика для лимитных ордеров с задержкой проверки
        """

        # Проверяем касание entry
        entry_hit = (low <= pos.entry <= high)

        if entry_hit and not pos.entry_touched:
            pos.entry_touched = True

        # ВАЖНО: НЕ ПРОВЕРЯЕМ условия отмены на первых N свечах
        MIN_CANDLES_BEFORE_CANCEL_CHECK = 3  # Минимум 3 свечи перед проверкой отмены

        if pos.candles_since_signal < MIN_CANDLES_BEFORE_CANCEL_CHECK:
            # На ранних свечах ТОЛЬКО открываем позицию при касании entry
            if entry_hit:
                pos.status = PositionStatus.OPEN
                pos.opened_at = ts

                if self.debug_mode:
                    print(f"[TRACKER] Position OPENED (early): {pos.symbol} {pos.direction} @ {pos.entry:.5f}")

                if self.publisher and hasattr(self.publisher, "publish_position_opened"):
                    try:
                        await self.publisher.publish_position_opened(pos)
                    except Exception as e:
                        print(f"Error publishing position opened: {e}")
            else:
                if self.debug_mode:
                    print(
                        f"[TRACKER] Waiting for entry, candle {pos.candles_since_signal}/{MIN_CANDLES_BEFORE_CANCEL_CHECK}")

            return  # ВЫХОДИМ, не проверяя условия отмены

        # === После N свечей начинаем проверять условия отмены ===

        # Проверяем достижение TP1 БЕЗ касания entry (цена перескочила)
        if pos.take_profits and not pos.entry_touched:
            tp1 = pos.take_profits[0]

            skip_detected = False
            reason = ""

            if pos.direction in ("LONG", "BUY"):
                # Для лонга: проверяем, не ушла ли цена к TP без касания entry
                if high >= tp1 and low > pos.entry:
                    skip_detected = True
                    reason = f"Price jumped over entry (Low: {low:.5f} > Entry: {pos.entry:.5f}) to TP1 {tp1:.5f}"
            else:  # SHORT/SELL
                # Для шорта: проверяем, не ушла ли цена к TP без касания entry
                if low <= tp1 and high < pos.entry:
                    skip_detected = True
                    reason = f"Price jumped under entry (High: {high:.5f} < Entry: {pos.entry:.5f}) to TP1 {tp1:.5f}"

            if skip_detected:
                pos.status = PositionStatus.CANCELLED
                pos.outcome = "MISSED"
                pos.cancelled_at = ts

                if self.debug_mode:
                    print(f"[TRACKER] Position CANCELLED: {reason}")

                if self.publisher and hasattr(self.publisher, "publish_position_cancelled"):
                    try:
                        await self.publisher.publish_position_cancelled(pos, reason=reason)
                    except Exception as e:
                        print(f"Error publishing cancellation: {e}")
                return

        # Если entry задет - открываем позицию
        if entry_hit and not pos.opened_at:
            pos.status = PositionStatus.OPEN
            pos.opened_at = ts

            if self.debug_mode:
                print(f"[TRACKER] Position OPENED: {pos.symbol} {pos.direction} @ {pos.entry:.5f}")

            if self.publisher and hasattr(self.publisher, "publish_position_opened"):
                try:
                    await self.publisher.publish_position_opened(pos)
                except Exception as e:
                    print(f"Error publishing position opened: {e}")

    async def _process_open(
            self,
            pos: TrackedPosition,
            high: float,
            low: float,
            close: float,
            ts: Optional[int],
    ) -> None:
        """
        ИСПРАВЛЕННАЯ логика для открытых позиций с задержкой проверки TP/SL
        """

        # ВАЖНО: Даем позиции "дышать" минимум 1 свечу после открытия
        MIN_CANDLES_AFTER_OPEN = 1

        if pos.opened_at and ts:
            # Считаем свечи с момента открытия
            candles_since_open = pos.candles_since_signal - (pos.candles_since_signal - 1)  # Упрощенно

            # Более точный расчет через время
            time_since_open = ts - pos.opened_at
            min_time_before_check = 60000  # 1 минута минимум

            if time_since_open < min_time_before_check:
                if self.debug_mode:
                    print(f"[TRACKER] Position just opened, skipping TP/SL check")
                return

        # Проверяем направление правильно
        is_long = pos.direction in ("LONG", "BUY")
        is_short = pos.direction in ("SHORT", "SELL")

        if not (is_long or is_short):
            print(f"⚠ Unknown direction in open position: {pos.direction}")
            return

        # ---------------- ПРИОРИТЕТ 1: Проверка SL ----------------
        sl_hit = False

        if is_long:
            # Для лонга SL должен быть ниже entry
            if pos.stop_loss >= pos.entry:
                print(f"⚠ Invalid LONG SL: {pos.stop_loss:.5f} >= Entry {pos.entry:.5f}")
            else:
                sl_hit = (low <= pos.stop_loss)
                if self.debug_mode and sl_hit:
                    print(f"[TRACKER] LONG SL hit: Low {low:.5f} <= SL {pos.stop_loss:.5f}")
        else:  # SHORT
            # Для шорта SL должен быть выше entry
            if pos.stop_loss <= pos.entry:
                print(f"⚠ Invalid SHORT SL: {pos.stop_loss:.5f} <= Entry {pos.entry:.5f}")
            else:
                sl_hit = (high >= pos.stop_loss)
                if self.debug_mode and sl_hit:
                    print(f"[TRACKER] SHORT SL hit: High {high:.5f} >= SL {pos.stop_loss:.5f}")

        if sl_hit:
            pos.status = PositionStatus.CLOSED
            pos.outcome = "SL"
            pos.closed_at = ts

            if self.debug_mode:
                print(f"[TRACKER] Position STOPPED OUT: {pos.symbol} {pos.direction}")

            if self.publisher and hasattr(self.publisher, "publish_position_closed"):
                try:
                    await self.publisher.publish_position_closed(pos, hit_tp_index=None)
                except Exception as e:
                    print(f"Error publishing SL: {e}")
            return

        # ---------------- ПРИОРИТЕТ 2: Проверка TP ----------------
        if not pos.take_profits:
            return

        for idx, tp in enumerate(pos.take_profits):
            tp_hit = False

            if is_long:
                # Для лонга TP должен быть выше entry
                if tp <= pos.entry:
                    print(f"⚠ Invalid LONG TP{idx + 1}: {tp:.5f} <= Entry {pos.entry:.5f}")
                    continue
                tp_hit = (high >= tp)
                if self.debug_mode and tp_hit:
                    print(f"[TRACKER] LONG TP{idx + 1} hit: High {high:.5f} >= TP {tp:.5f}")
            else:  # SHORT
                # Для шорта TP должен быть ниже entry
                if tp >= pos.entry:
                    print(f"⚠ Invalid SHORT TP{idx + 1}: {tp:.5f} >= Entry {pos.entry:.5f}")
                    continue
                tp_hit = (low <= tp)
                if self.debug_mode and tp_hit:
                    print(f"[TRACKER] SHORT TP{idx + 1} hit: Low {low:.5f} <= TP {tp:.5f}")

            if tp_hit:
                # Отмечаем какой TP достигнут
                pos.hit_tp_index = idx

                # УЛУЧШЕНИЕ: Частичное закрытие для TP1
                if idx == 0:  # TP1
                    # Можно реализовать частичное закрытие
                    # Пока закрываем полностью
                    pos.status = PositionStatus.CLOSED
                    pos.outcome = "TP"
                    pos.closed_at = ts

                    if self.debug_mode:
                        print(f"[TRACKER] TP1 HIT: {pos.symbol} {pos.direction}")
                else:
                    # TP2 - полное закрытие
                    pos.status = PositionStatus.CLOSED
                    pos.outcome = "TP"
                    pos.closed_at = ts

                    if self.debug_mode:
                        print(f"[TRACKER] TP2 HIT: {pos.symbol} {pos.direction}")

                if self.publisher and hasattr(self.publisher, "publish_position_closed"):
                    try:
                        await self.publisher.publish_position_closed(pos, hit_tp_index=idx)
                    except Exception as e:
                        print(f"Error publishing TP: {e}")

                # После первого достигнутого TP выходим
                break

    def cleanup_old_positions(self, max_age_ms: int = 86400000 * 7) -> int:
        """
        Удаляет позиции старше max_age_ms (по умолчанию 7 дней)
        Возвращает количество удаленных позиций
        """
        now = int(time.time() * 1000)
        removed = 0

        for symbol in list(self.positions.keys()):
            positions = self.positions[symbol]
            new_positions = []

            for pos in positions:
                age = now - pos.created_at
                # Удаляем только закрытые/отмененные старые позиции
                if age > max_age_ms and pos.status in (PositionStatus.CLOSED, PositionStatus.CANCELLED):
                    removed += 1
                else:
                    new_positions.append(pos)

            if new_positions:
                self.positions[symbol] = new_positions
            else:
                del self.positions[symbol]

        return removed

    def get_stats(self) -> Dict:
        """
        Возвращает статистику по позициям
        """
        stats = {
            "total": 0,
            "pending": 0,
            "open": 0,
            "closed_tp": 0,
            "closed_sl": 0,
            "cancelled": 0,
            "by_chain": {},
            "by_symbol": {}
        }

        for symbol, symbol_positions in self.positions.items():
            symbol_stats = {
                "total": 0,
                "pending": 0,
                "open": 0,
                "tp": 0,
                "sl": 0
            }

            for pos in symbol_positions:
                stats["total"] += 1
                symbol_stats["total"] += 1

                if pos.status == PositionStatus.PENDING:
                    stats["pending"] += 1
                    symbol_stats["pending"] += 1
                elif pos.status == PositionStatus.OPEN:
                    stats["open"] += 1
                    symbol_stats["open"] += 1
                elif pos.status == PositionStatus.CLOSED:
                    if pos.outcome == "TP":
                        stats["closed_tp"] += 1
                        symbol_stats["tp"] += 1
                    elif pos.outcome == "SL":
                        stats["closed_sl"] += 1
                        symbol_stats["sl"] += 1
                elif pos.status == PositionStatus.CANCELLED:
                    stats["cancelled"] += 1

                # Статистика по цепочкам
                if pos.chain_id not in stats["by_chain"]:
                    stats["by_chain"][pos.chain_id] = {
                        "total": 0, "tp": 0, "sl": 0, "cancelled": 0
                    }

                stats["by_chain"][pos.chain_id]["total"] += 1
                if pos.outcome == "TP":
                    stats["by_chain"][pos.chain_id]["tp"] += 1
                elif pos.outcome == "SL":
                    stats["by_chain"][pos.chain_id]["sl"] += 1
                elif pos.outcome == "MISSED":
                    stats["by_chain"][pos.chain_id]["cancelled"] += 1

            if symbol_stats["total"] > 0:
                stats["by_symbol"][symbol] = symbol_stats

        return stats

    def get_active_positions(self) -> List[TrackedPosition]:
        """
        Возвращает список всех активных (pending + open) позиций
        """
        active = []
        for symbol_positions in self.positions.values():
            for pos in symbol_positions:
                if pos.status in (PositionStatus.PENDING, PositionStatus.OPEN):
                    active.append(pos)
        return active