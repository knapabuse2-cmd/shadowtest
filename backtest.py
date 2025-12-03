# backtest_v2.py - БЭКТЕСТ С ВЫБОРОМ ПЕРИОДА И ДЕТАЛЬНОЙ СТАТИСТИКОЙ
# ===================================================================

import asyncio
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import json
import time
import csv
import os

import ccxt.async_support as ccxt

# === Детекторы ===
from analysis_detectors import (
    OrderBlockDetector,
    FairValueGapDetector,
    FractalDetector,
    VolumeContextBuilder,
)
from idm_detector import IDMDetector

# === Цепочки ===
from analysis_chains import (
    Chain_1_1,
    Chain_1_2,
    Chain_1_3,
    Chain_1_4,
    Chain_1_5,
    Chain_3_2,
    Signal_1,
    Chain_2_6,
)

from analysis_interfaces import ChainSignal, ChainContext, DetectionResult
from core_orchestrator import Orchestrator


# ================================
#   КОНФИГУРАЦИЯ
# ================================

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "NEAR/USDT", "ARB/USDT", "APT/USDT", "ATOM/USDT",
    "LTC/USDT", "BCH/USDT",
]

TIMEFRAMES = ["1d", "4h", "1h", "15m"]
BASE_TF = "15m"
WARMUP_BARS = 300

# Валидация сигналов
MIN_RR = 1.5
MIN_SL_PERCENT = 0.01
MAX_SL_PERCENT = 0.025

MAX_POSITIONS_PER_SYMBOL = 2
POSITION_TIMEOUT_BARS = 50


# ================================
#   МОДЕЛИ
# ================================

@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class PositionStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CANCELLED = "CANCELLED"


@dataclass
class BacktestPosition:
    symbol: str
    chain_id: str
    direction: str
    entry: float
    stop_loss: float
    take_profits: List[float]
    rr: float
    signal_bar_idx: int
    signal_time: datetime
    status: PositionStatus = PositionStatus.PENDING
    entry_bar_idx: Optional[int] = None
    entry_time: Optional[datetime] = None
    exit_bar_idx: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_r: float = 0.0
    bars_in_trade: int = 0
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "chain_id": self.chain_id,
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "tp1": self.take_profits[0] if self.take_profits else None,
            "rr": round(self.rr, 2),
            "status": self.status.value,
            "pnl_r": round(self.pnl_r, 2),
            "signal_time": self.signal_time.strftime("%Y-%m-%d %H:%M") if self.signal_time else None,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M") if self.entry_time else None,
            "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M") if self.exit_time else None,
            "exit_price": self.exit_price,
            "bars_in_trade": self.bars_in_trade,
        }


# ================================
#   ЗАГРУЗЧИК ДАННЫХ С BINANCE
# ================================

class BinanceHistoricalLoader:
    """Загружает реальные исторические данные с Binance"""
    
    def __init__(self):
        self.exchange = ccxt.binance({"enableRateLimit": True})
    
    async def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Candle]:
        """Загружает свечи за период"""
        
        candles = []
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Binance limit = 1000 свечей за запрос
        while since < end_ms:
            try:
                raw = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                if not raw:
                    break
                
                for t, o, h, l, c, v in raw:
                    if t > end_ms:
                        break
                    candles.append(Candle(
                        time=t,
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v)
                    ))
                
                # Следующий запрос с последней свечи
                since = raw[-1][0] + 1
                
                # Пауза чтобы не превысить лимит
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"    Error loading {symbol} {timeframe}: {e}")
                break
        
        return candles
    
    async def close(self):
        await self.exchange.close()


# ================================
#   ИСТОРИЧЕСКИЙ DATA SOURCE
# ================================

class HistoricalDataSource:
    def __init__(self, all_candles: Dict[str, Dict[str, List[Candle]]]):
        self.all_candles = all_candles
        self.current_bar_idx: Dict[str, int] = {}
    
    def set_current_bar(self, symbol: str, bar_idx: int):
        self.current_bar_idx[symbol] = bar_idx
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> List[Candle]:
        if symbol not in self.all_candles:
            return []
        
        full = self.all_candles[symbol].get(timeframe, [])
        if not full:
            return []
        
        bar_idx = self.current_bar_idx.get(symbol, len(full) - 1)
        
        if timeframe == BASE_TF:
            end_idx = min(bar_idx + 1, len(full))
            start_idx = max(0, end_idx - limit)
            return full[start_idx:end_idx]
        
        base_candles = self.all_candles[symbol].get(BASE_TF, [])
        if bar_idx < len(base_candles):
            current_time = base_candles[bar_idx].time
            partial = [c for c in full if c.time <= current_time]
            return partial[-limit:] if partial else []
        
        return full[-limit:]


# ================================
#   ВАЛИДАТОР
# ================================

def validate_signal(signal: ChainSignal) -> Tuple[bool, str]:
    if signal.rr < MIN_RR:
        return False, f"RR {signal.rr:.2f} < {MIN_RR}"
    
    direction = str(signal.direction).upper().replace("DIRECTION.", "")
    
    if "LONG" in direction:
        if signal.stop_loss >= signal.entry:
            return False, "LONG SL >= Entry"
    elif "SHORT" in direction:
        if signal.stop_loss <= signal.entry:
            return False, "SHORT SL <= Entry"
    
    sl_distance = abs(signal.entry - signal.stop_loss)
    sl_percent = sl_distance / signal.entry if signal.entry != 0 else 0
    
    if sl_percent < MIN_SL_PERCENT:
        return False, f"SL {sl_percent:.2%} < {MIN_SL_PERCENT:.0%}"
    if sl_percent > MAX_SL_PERCENT:
        return False, f"SL {sl_percent:.2%} > {MAX_SL_PERCENT:.1%}"
    if not signal.take_profits:
        return False, "No TPs"
    
    return True, "OK"


# ================================
#   СИМУЛЯЦИЯ ПОЗИЦИЙ
# ================================

def check_pending_entry(pos: BacktestPosition, bar: Candle) -> bool:
    return bar.low <= pos.entry <= bar.high


def check_open_position(pos: BacktestPosition, bar: Candle) -> Optional[Tuple[str, float]]:
    is_long = pos.direction in ("LONG", "BUY")
    
    # SL первый (приоритет)
    if is_long:
        if bar.low <= pos.stop_loss:
            return ("SL", pos.stop_loss)
    else:
        if bar.high >= pos.stop_loss:
            return ("SL", pos.stop_loss)
    
    # TP
    for tp in pos.take_profits:
        if is_long:
            if bar.high >= tp:
                return ("TP", tp)
        else:
            if bar.low <= tp:
                return ("TP", tp)
    
    return None


def check_skip_entry(pos: BacktestPosition, bar: Candle) -> bool:
    if not pos.take_profits:
        return False
    
    tp1 = pos.take_profits[0]
    is_long = pos.direction in ("LONG", "BUY")
    
    if is_long:
        if bar.low > pos.entry and bar.high >= tp1:
            return True
    else:
        if bar.high < pos.entry and bar.low <= tp1:
            return True
    
    return False


# ================================
#   БЭКТЕСТ СИМВОЛА
# ================================

async def backtest_symbol(
    symbol: str,
    all_candles: Dict[str, Dict[str, List[Candle]]],
    verbose: bool = False
) -> List[BacktestPosition]:
    
    base_candles = all_candles[symbol].get(BASE_TF, [])
    if len(base_candles) <= WARMUP_BARS + 10:
        print(f"  ⚠ {symbol}: only {len(base_candles)} bars, need {WARMUP_BARS + 10}+")
        return []
    
    start_time_str = datetime.utcfromtimestamp(base_candles[0].time / 1000).strftime("%Y-%m-%d")
    end_time_str = datetime.utcfromtimestamp(base_candles[-1].time / 1000).strftime("%Y-%m-%d")
    print(f"\n📊 {symbol}: {len(base_candles)} bars ({start_time_str} → {end_time_str})")
    
    hist_source = HistoricalDataSource(all_candles)
    
    detectors = {
        "OrderBlock": OrderBlockDetector(),
        "FairValueGap": FairValueGapDetector(),
        "Fractal": FractalDetector(),
        "VolumeContext": VolumeContextBuilder(),
        "IDM": IDMDetector(),
    }
    
    chains = [
        Chain_1_1(), Chain_1_2(), Chain_1_3(), Chain_1_4(), Chain_1_5(),
        Chain_3_2(), Signal_1(), Chain_2_6(),
    ]
    
    orchestrator = Orchestrator(hist_source, detectors, chains)
    orchestrator.set_logger(None, verbose=False)
    
    pending: List[BacktestPosition] = []
    open_pos: List[BacktestPosition] = []
    closed: List[BacktestPosition] = []
    
    seen: Set[str] = set()
    total_signals = 0
    validated = 0
    
    total_bars = len(base_candles)
    
    for bar_idx in range(WARMUP_BARS, total_bars):
        bar = base_candles[bar_idx]
        bar_time = datetime.utcfromtimestamp(bar.time / 1000)
        hist_source.set_current_bar(symbol, bar_idx)
        
        # 1. Pending позиции
        still_pending = []
        for pos in pending:
            if bar_idx - pos.signal_bar_idx > POSITION_TIMEOUT_BARS:
                pos.status = PositionStatus.CANCELLED
                pos.exit_bar_idx = bar_idx
                pos.exit_time = bar_time
                closed.append(pos)
                continue
            
            if check_skip_entry(pos, bar):
                pos.status = PositionStatus.CANCELLED
                pos.exit_bar_idx = bar_idx
                pos.exit_time = bar_time
                closed.append(pos)
                continue
            
            if check_pending_entry(pos, bar):
                pos.status = PositionStatus.OPEN
                pos.entry_bar_idx = bar_idx
                pos.entry_time = bar_time
                open_pos.append(pos)
                if verbose:
                    print(f"    ✅ ENTRY {pos.chain_id} {pos.direction} @ {pos.entry:.2f}")
            else:
                still_pending.append(pos)
        
        pending = still_pending
        
        # 2. Открытые позиции
        still_open = []
        for pos in open_pos:
            pos.bars_in_trade += 1
            result = check_open_position(pos, bar)
            
            if result:
                outcome, exit_price = result
                pos.exit_bar_idx = bar_idx
                pos.exit_time = bar_time
                pos.exit_price = exit_price
                
                if outcome == "SL":
                    pos.status = PositionStatus.CLOSED_SL
                    pos.pnl_r = -1.0
                else:
                    pos.status = PositionStatus.CLOSED_TP
                    pos.pnl_r = pos.rr
                
                closed.append(pos)
                if verbose:
                    emoji = "❌" if outcome == "SL" else "✅"
                    print(f"    {emoji} {outcome} {pos.chain_id} ({pos.pnl_r:+.2f}R) after {pos.bars_in_trade} bars")
            else:
                still_open.append(pos)
        
        open_pos = still_open
        
        # 3. Новые сигналы
        if len(pending) + len(open_pos) >= MAX_POSITIONS_PER_SYMBOL:
            continue
        
        try:
            signals = await orchestrator.analyze_symbol(symbol)
            
            for sig in signals:
                total_signals += 1
                
                sig_hash = f"{sig.chain_id}_{sig.entry:.4f}_{bar_idx // 10}"
                if sig_hash in seen:
                    continue
                seen.add(sig_hash)
                
                is_valid, reason = validate_signal(sig)
                if not is_valid:
                    if verbose:
                        print(f"    ✗ {sig.chain_id}: {reason}")
                    continue
                
                validated += 1
                direction = str(sig.direction).upper().replace("DIRECTION.", "")
                
                pos = BacktestPosition(
                    symbol=symbol,
                    chain_id=sig.chain_id,
                    direction=direction,
                    entry=sig.entry,
                    stop_loss=sig.stop_loss,
                    take_profits=list(sig.take_profits),
                    rr=sig.rr,
                    signal_bar_idx=bar_idx,
                    signal_time=bar_time
                )
                pending.append(pos)
                
                if verbose:
                    print(f"    📊 SIGNAL {sig.chain_id} {direction} @ {sig.entry:.2f} (RR={sig.rr:.2f})")
                    
        except Exception as e:
            if verbose:
                print(f"    ⚠ Error: {e}")
        
        # Прогресс
        if bar_idx % 200 == 0:
            progress = (bar_idx - WARMUP_BARS) / (total_bars - WARMUP_BARS) * 100
            finished = len([p for p in closed if p.status in (PositionStatus.CLOSED_TP, PositionStatus.CLOSED_SL)])
            print(f"  {progress:.0f}% | Signals: {validated} | Finished: {finished} | P:{len(pending)} O:{len(open_pos)}")
    
    # Закрываем оставшиеся
    final_time = datetime.utcfromtimestamp(base_candles[-1].time / 1000)
    for pos in pending:
        pos.status = PositionStatus.CANCELLED
        pos.exit_bar_idx = total_bars - 1
        pos.exit_time = final_time
        closed.append(pos)
    
    for pos in open_pos:
        pos.exit_bar_idx = total_bars - 1
        pos.exit_time = final_time
        # Оставляем OPEN статус
        closed.append(pos)
    
    finished = [p for p in closed if p.status in (PositionStatus.CLOSED_TP, PositionStatus.CLOSED_SL)]
    print(f"  ✅ {symbol}: {len(finished)} trades | {total_signals} raw signals | {validated} validated")
    
    return closed


# ================================
#   ЗАГРУЗКА ДАННЫХ
# ================================

async def load_all_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Dict[str, List[Candle]]]:
    """Загружает все данные с Binance"""
    
    print(f"\n📥 Loading data: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"   Symbols: {len(symbols)}, Timeframes: {TIMEFRAMES}")
    
    loader = BinanceHistoricalLoader()
    all_candles: Dict[str, Dict[str, List[Candle]]] = {}
    
    for sym in symbols:
        print(f"\n  {sym}:", end="")
        all_candles[sym] = {}
        
        for tf in TIMEFRAMES:
            candles = await loader.load_candles(sym, tf, start_date, end_date)
            all_candles[sym][tf] = candles
            print(f" {tf}({len(candles)})", end="")
        
        print()
    
    await loader.close()
    return all_candles


# ================================
#   ДЕТАЛЬНАЯ СТАТИСТИКА
# ================================

def build_detailed_report(
    all_positions: List[BacktestPosition],
    start_date: datetime,
    end_date: datetime
) -> Dict:
    """Строит детальный отчёт"""
    
    print("\n" + "=" * 70)
    print("📊 ДЕТАЛЬНЫЙ ОТЧЁТ ПО БЭКТЕСТУ")
    print("=" * 70)
    print(f"Период: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    if not all_positions:
        print("Нет позиций.")
        return {}
    
    # Разделяем по статусам
    finished = [p for p in all_positions if p.status in (PositionStatus.CLOSED_TP, PositionStatus.CLOSED_SL)]
    cancelled = [p for p in all_positions if p.status == PositionStatus.CANCELLED]
    still_open = [p for p in all_positions if p.status == PositionStatus.OPEN]
    
    if not finished:
        print(f"\nНет завершённых сделок.")
        print(f"  Отменённых: {len(cancelled)}")
        print(f"  Ещё открытых: {len(still_open)}")
        return {}
    
    # === ОСНОВНАЯ СТАТИСТИКА ===
    wins = [p for p in finished if p.status == PositionStatus.CLOSED_TP]
    losses = [p for p in finished if p.status == PositionStatus.CLOSED_SL]
    
    total = len(finished)
    win_count = len(wins)
    loss_count = len(losses)
    winrate = win_count / total * 100
    
    total_pnl = sum(p.pnl_r for p in finished)
    avg_pnl = total_pnl / total
    
    gross_profit = sum(p.pnl_r for p in wins)
    gross_loss = abs(sum(p.pnl_r for p in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Средний RR
    avg_rr_wins = sum(p.rr for p in wins) / len(wins) if wins else 0
    
    # Просадка и эквити
    equity_curve = []
    equity = 0.0
    max_equity = 0.0
    max_drawdown = 0.0
    max_dd_percent = 0.0
    
    for p in finished:
        equity += p.pnl_r
        equity_curve.append(equity)
        if equity > max_equity:
            max_equity = equity
        dd = max_equity - equity
        if dd > max_drawdown:
            max_drawdown = dd
        if max_equity > 0:
            dd_pct = dd / max_equity * 100
            if dd_pct > max_dd_percent:
                max_dd_percent = dd_pct
    
    print(f"\n{'─' * 70}")
    print("📈 ОСНОВНАЯ СТАТИСТИКА")
    print(f"{'─' * 70}")
    print(f"  Всего сделок:     {total}")
    print(f"  Выигрышных:       {win_count} ({winrate:.1f}%)")
    print(f"  Проигрышных:      {loss_count} ({100 - winrate:.1f}%)")
    print(f"  Отменённых:       {len(cancelled)}")
    print(f"  Ещё открытых:     {len(still_open)}")
    
    print(f"\n{'─' * 70}")
    print("💰 ФИНАНСОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'─' * 70}")
    print(f"  Общий PnL:        {total_pnl:+.2f} R")
    print(f"  Средний PnL:      {avg_pnl:+.3f} R")
    print(f"  Gross Profit:     {gross_profit:+.2f} R")
    print(f"  Gross Loss:       {-gross_loss:.2f} R")
    print(f"  Profit Factor:    {profit_factor:.2f}")
    print(f"  Max Drawdown:     {max_drawdown:.2f} R ({max_dd_percent:.1f}%)")
    print(f"  Средний RR побед: {avg_rr_wins:.2f}")
    
    # === СЕРИИ ===
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0
    
    for p in finished:
        if p.status == PositionStatus.CLOSED_TP:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    
    print(f"\n{'─' * 70}")
    print("📊 СЕРИИ")
    print(f"{'─' * 70}")
    print(f"  Макс. серия побед:    {max_win_streak}")
    print(f"  Макс. серия потерь:   {max_loss_streak}")
    
    # === ДЛИТЕЛЬНОСТЬ СДЕЛОК ===
    durations = [p.bars_in_trade for p in finished if p.bars_in_trade > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        win_durations = [p.bars_in_trade for p in wins if p.bars_in_trade > 0]
        loss_durations = [p.bars_in_trade for p in losses if p.bars_in_trade > 0]
        
        avg_win_duration = sum(win_durations) / len(win_durations) if win_durations else 0
        avg_loss_duration = sum(loss_durations) / len(loss_durations) if loss_durations else 0
        
        print(f"\n{'─' * 70}")
        print("⏱ ДЛИТЕЛЬНОСТЬ СДЕЛОК (в барах 15m)")
        print(f"{'─' * 70}")
        print(f"  Средняя:          {avg_duration:.1f} баров (~{avg_duration * 15 / 60:.1f} часов)")
        print(f"  Мин/Макс:         {min_duration} / {max_duration} баров")
        print(f"  Средняя для TP:   {avg_win_duration:.1f} баров")
        print(f"  Средняя для SL:   {avg_loss_duration:.1f} баров")
    
    # === ПО ЦЕПОЧКАМ ===
    print(f"\n{'─' * 70}")
    print("🔗 СТАТИСТИКА ПО ЦЕПОЧКАМ")
    print(f"{'─' * 70}")
    
    chain_stats = {}
    for p in finished:
        if p.chain_id not in chain_stats:
            chain_stats[p.chain_id] = {"wins": 0, "losses": 0, "pnl": 0.0, "rr_sum": 0.0}
        
        if p.status == PositionStatus.CLOSED_TP:
            chain_stats[p.chain_id]["wins"] += 1
            chain_stats[p.chain_id]["rr_sum"] += p.rr
        else:
            chain_stats[p.chain_id]["losses"] += 1
        chain_stats[p.chain_id]["pnl"] += p.pnl_r
    
    for chain_id, s in sorted(chain_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100 if t > 0 else 0
        avg_rr = s["rr_sum"] / s["wins"] if s["wins"] > 0 else 0
        emoji = "🟢" if s["pnl"] > 0 else "🔴"
        print(f"  {emoji} {chain_id:8s} | {s['wins']:2d}W / {s['losses']:2d}L | WR: {wr:5.1f}% | PnL: {s['pnl']:+6.2f}R | AvgRR: {avg_rr:.2f}")
    
    # === ПО СИМВОЛАМ ===
    print(f"\n{'─' * 70}")
    print("💹 СТАТИСТИКА ПО СИМВОЛАМ")
    print(f"{'─' * 70}")
    
    symbol_stats = {}
    for p in finished:
        if p.symbol not in symbol_stats:
            symbol_stats[p.symbol] = {"wins": 0, "losses": 0, "pnl": 0.0}
        
        if p.status == PositionStatus.CLOSED_TP:
            symbol_stats[p.symbol]["wins"] += 1
        else:
            symbol_stats[p.symbol]["losses"] += 1
        symbol_stats[p.symbol]["pnl"] += p.pnl_r
    
    for sym, s in sorted(symbol_stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100 if t > 0 else 0
        emoji = "🟢" if s["pnl"] > 0 else "🔴"
        print(f"  {emoji} {sym:12s} | {s['wins']:2d}W / {s['losses']:2d}L | WR: {wr:5.1f}% | PnL: {s['pnl']:+6.2f}R")
    
    # === ПО НАПРАВЛЕНИЮ ===
    print(f"\n{'─' * 70}")
    print("📊 ПО НАПРАВЛЕНИЮ")
    print(f"{'─' * 70}")
    
    for direction in ["LONG", "SHORT"]:
        dir_trades = [p for p in finished if p.direction == direction]
        if dir_trades:
            dir_wins = len([p for p in dir_trades if p.status == PositionStatus.CLOSED_TP])
            dir_total = len(dir_trades)
            dir_wr = dir_wins / dir_total * 100
            dir_pnl = sum(p.pnl_r for p in dir_trades)
            emoji = "🟢" if dir_pnl > 0 else "🔴"
            print(f"  {emoji} {direction:6s} | {dir_wins:2d}W / {dir_total - dir_wins:2d}L | WR: {dir_wr:5.1f}% | PnL: {dir_pnl:+6.2f}R")
    
    # === ПО ДНЯМ НЕДЕЛИ ===
    print(f"\n{'─' * 70}")
    print("📅 ПО ДНЯМ НЕДЕЛИ (сигнал)")
    print(f"{'─' * 70}")
    
    weekday_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    weekday_stats = {i: {"wins": 0, "losses": 0, "pnl": 0.0} for i in range(7)}
    
    for p in finished:
        if p.signal_time:
            wd = p.signal_time.weekday()
            if p.status == PositionStatus.CLOSED_TP:
                weekday_stats[wd]["wins"] += 1
            else:
                weekday_stats[wd]["losses"] += 1
            weekday_stats[wd]["pnl"] += p.pnl_r
    
    for wd in range(7):
        s = weekday_stats[wd]
        t = s["wins"] + s["losses"]
        if t > 0:
            wr = s["wins"] / t * 100
            emoji = "🟢" if s["pnl"] > 0 else "🔴"
            print(f"  {emoji} {weekday_names[wd]:2s} | {s['wins']:2d}W / {s['losses']:2d}L | WR: {wr:5.1f}% | PnL: {s['pnl']:+6.2f}R")
    
    # === ПО ЧАСАМ ===
    print(f"\n{'─' * 70}")
    print("🕐 ПО ЧАСАМ UTC (сигнал)")
    print(f"{'─' * 70}")
    
    hour_stats = {i: {"wins": 0, "losses": 0, "pnl": 0.0} for i in range(24)}
    
    for p in finished:
        if p.signal_time:
            h = p.signal_time.hour
            if p.status == PositionStatus.CLOSED_TP:
                hour_stats[h]["wins"] += 1
            else:
                hour_stats[h]["losses"] += 1
            hour_stats[h]["pnl"] += p.pnl_r
    
    # Показываем только часы с активностью
    active_hours = [(h, s) for h, s in hour_stats.items() if s["wins"] + s["losses"] > 0]
    active_hours.sort(key=lambda x: x[1]["pnl"], reverse=True)
    
    for h, s in active_hours[:12]:  # Топ 12 часов
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100
        emoji = "🟢" if s["pnl"] > 0 else "🔴"
        print(f"  {emoji} {h:02d}:00 | {s['wins']:2d}W / {s['losses']:2d}L | WR: {wr:5.1f}% | PnL: {s['pnl']:+6.2f}R")
    
    # === ЛУЧШИЕ/ХУДШИЕ СДЕЛКИ ===
    print(f"\n{'─' * 70}")
    print("🏆 ТОП-5 ЛУЧШИХ СДЕЛОК")
    print(f"{'─' * 70}")
    
    best = sorted(wins, key=lambda p: p.rr, reverse=True)[:5]
    for p in best:
        print(f"  ✅ {p.symbol:12s} | {p.chain_id:8s} | {p.direction:5s} | RR: {p.rr:.2f} | {p.signal_time.strftime('%Y-%m-%d %H:%M') if p.signal_time else 'N/A'}")
    
    print(f"\n{'─' * 70}")
    print("💀 ПОСЛЕДНИЕ 5 ПРОИГРЫШНЫХ")
    print(f"{'─' * 70}")
    
    worst = losses[-5:] if len(losses) >= 5 else losses
    for p in worst:
        print(f"  ❌ {p.symbol:12s} | {p.chain_id:8s} | {p.direction:5s} | RR: {p.rr:.2f} | {p.signal_time.strftime('%Y-%m-%d %H:%M') if p.signal_time else 'N/A'}")
    
    print("\n" + "=" * 70)
    
    return {
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "summary": {
            "total": total,
            "wins": win_count,
            "losses": loss_count,
            "winrate": round(winrate, 2),
            "total_pnl": round(total_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "avg_rr_wins": round(avg_rr_wins, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        },
        "by_chain": chain_stats,
        "by_symbol": symbol_stats,
        "equity_curve": equity_curve,
    }


# ================================
#   СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ================================

def save_results(
    positions: List[BacktestPosition],
    report: Dict,
    output_dir: str = "."
):
    """Сохраняет результаты в JSON и CSV"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON с полным отчётом
    json_file = os.path.join(output_dir, f"backtest_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "report": report,
            "trades": [p.to_dict() for p in positions]
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON: {json_file}")
    
    # CSV со сделками
    csv_file = os.path.join(output_dir, f"backtest_trades_{timestamp}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if positions:
            writer = csv.DictWriter(f, fieldnames=positions[0].to_dict().keys())
            writer.writeheader()
            for p in positions:
                writer.writerow(p.to_dict())
    print(f"💾 CSV:  {csv_file}")


# ================================
#   MAIN
# ================================

async def main():
    parser = argparse.ArgumentParser(description='ICT Backtest v2')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--days', type=int, help='Number of days back from today', default=90)
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols', default=None)
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, help='Output directory', default='.')
    
    args = parser.parse_args()
    
    # Определяем период
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.utcnow()
    else:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
    
    # Символы
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        symbols = [s if '/' in s else f"{s}/USDT" for s in symbols]
    else:
        symbols = SYMBOLS
    
    print("=" * 70)
    print("🚀 ICT BACKTEST v2")
    print("=" * 70)
    print(f"\n📅 Период: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Символы: {', '.join(symbols)}")
    print(f"⚙️  Min RR: {MIN_RR}, SL: {MIN_SL_PERCENT*100:.0f}%-{MAX_SL_PERCENT*100:.1f}%")
    
    start_time = time.time()
    
    # 1. Загружаем данные
    all_candles = await load_all_data(symbols, start_date, end_date)
    
    # 2. Бэктест
    all_positions: List[BacktestPosition] = []
    
    for i, sym in enumerate(symbols):
        if sym not in all_candles or not all_candles[sym].get(BASE_TF):
            print(f"  ⚠ {sym}: No data")
            continue
        
        try:
            positions = await backtest_symbol(
                sym, 
                all_candles, 
                verbose=(args.verbose or i == 0)  # Первый символ всегда verbose
            )
            all_positions.extend(positions)
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Отчёт
    report = build_detailed_report(all_positions, start_date, end_date)
    
    # 4. Сохранение
    if report:
        save_results(all_positions, report, args.output)
    
    elapsed = time.time() - start_time
    print(f"\n⏱ Время: {elapsed:.1f}s")
    print("✅ ГОТОВО")


if __name__ == "__main__":
    asyncio.run(main())