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
    PARTIAL = "PARTIAL"  # TP1 взят, ждём TP2 или БУ
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
    outcome: Optional[str] = None  # "TP1" / "TP2" / "SL" / "BE" / "MISSED"
    hit_tp_index: Optional[int] = None
    last_price: Optional[float] = None
    last_update: Optional[int] = None
    candles_since_signal: int = 0
    entry_touched: bool = False
    # НОВЫЕ ПОЛЯ:
    signal_message_id: Optional[int] = None  # ID сообщения для reply
    original_stop_loss: Optional[float] = None  # Оригинальный SL для расчета RR
    at_breakeven: bool = False  # Позиция в безубытке
    tp1_hit: bool = False  # TP1 достигнут, ждём TP2
    realized_rr: float = 0.0  # Реализованный RR (накопительный)


class PositionTracker:
    """
    Отслеживает, что произошло с сигналами:
    - лимитка (pending)
    - вход (fill)
    - тейки/стоп/безубыток
    - отмена, если цена ушла в TP1 без входа
    """

    def __init__(self, publisher):
        self.publisher = publisher
        self.positions: Dict[str, List[TrackedPosition]] = {}
        self.debug_mode = False
        self.min_candles_before_check = 2
        # Статистика RR
        self.total_realized_rr: float = 0.0

    def register_signal(self, signal: ChainSignal, now_time: int, message_id: Optional[int] = None) -> None:
        """
        Вызывается каждый раз, когда бот нашёл новый сигнал и отправил его в канал.
        """
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
            last_update=now_time,
            signal_message_id=message_id,  # Сохраняем ID сообщения
            original_stop_loss=float(signal.stop_loss),  # Сохраняем оригинальный SL
        )

        self.positions.setdefault(pos.symbol, []).append(pos)

        if self.debug_mode:
            print(
                f"[TRACKER] Registered {pos.symbol} {pos.direction}: Entry={pos.entry:.5f}, SL={pos.stop_loss:.5f}, TPs={pos.take_profits}")

    async def update_with_candle(self, symbol: str, candle: Candle) -> None:
        """
        Обновляет статус позиций на основе новой свечи
        """
        if symbol not in self.positions:
            return

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
            if pos.status in (PositionStatus.CLOSED, PositionStatus.CANCELLED):
                continue

            # Обновляем счетчик свечей
            if pos.last_update is None:
                pos.last_update = ts
                pos.candles_since_signal = 0
            else:
                time_diff = abs(ts - pos.last_update)
                min_candle_interval = 60000

                if time_diff > min_candle_interval:
                    pos.candles_since_signal += 1
                    pos.last_update = ts

                    if self.debug_mode:
                        print(f"[TRACKER] {pos.symbol}: Candle #{pos.candles_since_signal} since signal")

            pos.last_price = close

            if pos.status == PositionStatus.PENDING:
                await self._process_pending(pos, high, low, close, ts)
            elif pos.status in (PositionStatus.OPEN, PositionStatus.PARTIAL):
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
        Логика для лимитных ордеров
        """
        entry_hit = (low <= pos.entry <= high)

        if entry_hit and not pos.entry_touched:
            pos.entry_touched = True

        MIN_CANDLES_BEFORE_CANCEL_CHECK = 3

        if pos.candles_since_signal < MIN_CANDLES_BEFORE_CANCEL_CHECK:
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
            return

        # После N свечей проверяем условия отмены
        if pos.take_profits and not pos.entry_touched:
            tp1 = pos.take_profits[0]

            skip_detected = False
            reason = ""

            if pos.direction in ("LONG", "BUY"):
                if high >= tp1 and low > pos.entry:
                    skip_detected = True
                    reason = f"Price jumped over entry to TP1"
            else:
                if low <= tp1 and high < pos.entry:
                    skip_detected = True
                    reason = f"Price jumped under entry to TP1"

            if skip_detected:
                pos.status = PositionStatus.CANCELLED
                pos.outcome = "MISSED"
                pos.cancelled_at = ts

                if self.publisher and hasattr(self.publisher, "publish_position_cancelled"):
                    try:
                        await self.publisher.publish_position_cancelled(pos, reason=reason)
                    except Exception as e:
                        print(f"Error publishing cancellation: {e}")
                return

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
        Логика для открытых позиций с поддержкой TP2 и БУ
        """
        MIN_CANDLES_AFTER_OPEN = 1

        if pos.opened_at and ts:
            time_since_open = ts - pos.opened_at
            min_time_before_check = 60000

            if time_since_open < min_time_before_check:
                if self.debug_mode:
                    print(f"[TRACKER] Position just opened, skipping TP/SL check")
                return

        is_long = pos.direction in ("LONG", "BUY")
        is_short = pos.direction in ("SHORT", "SELL")

        if not (is_long or is_short):
            print(f"⚠ Unknown direction in open position: {pos.direction}")
            return

        # Используем оригинальный SL для расчета RR
        original_risk = abs(pos.entry - pos.original_stop_loss) if pos.original_stop_loss else abs(pos.entry - pos.stop_loss)

        # ---------------- ПРИОРИТЕТ 1: Проверка SL/БУ ----------------
        sl_hit = False
        be_hit = False

        if is_long:
            if pos.stop_loss < pos.entry:
                sl_hit = (low <= pos.stop_loss)
            # Проверка БУ (если позиция в partial и SL = entry)
            if pos.at_breakeven and pos.tp1_hit:
                be_hit = (low <= pos.entry)
        else:
            if pos.stop_loss > pos.entry:
                sl_hit = (high >= pos.stop_loss)
            if pos.at_breakeven and pos.tp1_hit:
                be_hit = (high >= pos.entry)

        # Безубыток сработал
        if be_hit and pos.tp1_hit:
            pos.status = PositionStatus.CLOSED
            pos.outcome = "BE"
            pos.closed_at = ts
            # RR от TP1 уже учтён, БУ = 0
            
            if self.debug_mode:
                print(f"[TRACKER] Position hit BREAKEVEN: {pos.symbol}")

            if self.publisher and hasattr(self.publisher, "publish_position_breakeven"):
                try:
                    await self.publisher.publish_position_breakeven(pos)
                except Exception as e:
                    print(f"Error publishing BE: {e}")
            return

        # Стоп сработал
        if sl_hit and not pos.tp1_hit:
            pos.status = PositionStatus.CLOSED
            pos.outcome = "SL"
            pos.closed_at = ts
            pos.realized_rr = -1.0
            self.total_realized_rr -= 1.0

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
            # Пропускаем уже взятые TP
            if idx == 0 and pos.tp1_hit:
                continue

            tp_hit = False

            if is_long:
                if tp > pos.entry:
                    tp_hit = (high >= tp)
            else:
                if tp < pos.entry:
                    tp_hit = (low <= tp)

            if tp_hit:
                pos.hit_tp_index = idx

                if idx == 0:  # TP1
                    # Частичное закрытие - переход в PARTIAL статус
                    pos.tp1_hit = True
                    pos.status = PositionStatus.PARTIAL
                    pos.at_breakeven = True
                    
                    # Рассчитываем RR для TP1
                    tp1_distance = abs(tp - pos.entry)
                    tp1_rr = tp1_distance / original_risk if original_risk > 0 else 0
                    # Учитываем 50% позиции
                    pos.realized_rr = tp1_rr * 0.5
                    self.total_realized_rr += pos.realized_rr

                    # Переносим SL в БУ
                    pos.stop_loss = pos.entry

                    if self.debug_mode:
                        print(f"[TRACKER] TP1 HIT: {pos.symbol}, moving to breakeven, RR: +{tp1_rr:.2f} (50%)")

                    if self.publisher and hasattr(self.publisher, "publish_tp1_hit"):
                        try:
                            await self.publisher.publish_tp1_hit(pos, tp1_rr)
                        except Exception as e:
                            print(f"Error publishing TP1: {e}")

                elif idx == 1:  # TP2
                    pos.status = PositionStatus.CLOSED
                    pos.outcome = "TP2"
                    pos.closed_at = ts

                    # Рассчитываем RR для TP2 (оставшиеся 50%)
                    tp2_distance = abs(tp - pos.entry)
                    tp2_rr = tp2_distance / original_risk if original_risk > 0 else 0
                    # Добавляем 50% от TP2
                    additional_rr = tp2_rr * 0.5
                    pos.realized_rr += additional_rr
                    self.total_realized_rr += additional_rr

                    if self.debug_mode:
                        print(f"[TRACKER] TP2 HIT: {pos.symbol}, total RR: +{pos.realized_rr:.2f}")

                    if self.publisher and hasattr(self.publisher, "publish_position_closed"):
                        try:
                            await self.publisher.publish_position_closed(pos, hit_tp_index=idx)
                        except Exception as e:
                            print(f"Error publishing TP2: {e}")

                break  # После обработки TP выходим

    def cleanup_old_positions(self, max_age_ms: int = 86400000 * 7) -> int:
        """
        Удаляет позиции старше max_age_ms (по умолчанию 7 дней)
        """
        now = int(time.time() * 1000)
        removed = 0

        for symbol in list(self.positions.keys()):
            positions = self.positions[symbol]
            new_positions = []

            for pos in positions:
                age = now - pos.created_at
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
        Возвращает статистику по позициям включая RR
        """
        stats = {
            "total": 0,
            "pending": 0,
            "open": 0,
            "partial": 0,  # Позиции после TP1, ждут TP2
            "closed_tp": 0,
            "closed_tp1_only": 0,  # Закрыты по БУ после TP1
            "closed_tp2": 0,
            "closed_sl": 0,
            "cancelled": 0,
            "total_rr": self.total_realized_rr,
            "by_chain": {},
            "by_symbol": {}
        }

        for symbol, symbol_positions in self.positions.items():
            symbol_stats = {
                "total": 0,
                "pending": 0,
                "open": 0,
                "partial": 0,
                "tp": 0,
                "sl": 0,
                "rr": 0.0
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
                elif pos.status == PositionStatus.PARTIAL:
                    stats["partial"] += 1
                    symbol_stats["partial"] += 1
                elif pos.status == PositionStatus.CLOSED:
                    if pos.outcome in ("TP1", "TP2", "BE"):
                        stats["closed_tp"] += 1
                        symbol_stats["tp"] += 1
                        if pos.outcome == "TP2":
                            stats["closed_tp2"] += 1
                        elif pos.outcome == "BE":
                            stats["closed_tp1_only"] += 1
                    elif pos.outcome == "SL":
                        stats["closed_sl"] += 1
                        symbol_stats["sl"] += 1
                elif pos.status == PositionStatus.CANCELLED:
                    stats["cancelled"] += 1

                symbol_stats["rr"] += pos.realized_rr

                # Статистика по цепочкам
                if pos.chain_id not in stats["by_chain"]:
                    stats["by_chain"][pos.chain_id] = {
                        "total": 0, "tp": 0, "sl": 0, "cancelled": 0, "rr": 0.0
                    }

                stats["by_chain"][pos.chain_id]["total"] += 1
                stats["by_chain"][pos.chain_id]["rr"] += pos.realized_rr
                
                if pos.outcome in ("TP1", "TP2", "BE"):
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
        Возвращает список всех активных позиций
        """
        active = []
        for symbol_positions in self.positions.values():
            for pos in symbol_positions:
                if pos.status in (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.PARTIAL):
                    active.append(pos)
        return active