# config_and_main.py - ИСПРАВЛЕННАЯ ВЕРСИЯ

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from bias_detector import ICTBiasDetector, BiasContext, MarketBias
from data.binance_source import BinanceDataSource
from output_module import TelegramSignalPublisher
from output_log_publisher import TelegramLogPublisher
from core_orchestrator import Orchestrator
from breaker_detector import BreakerDetector
from mitigation_block_detector import MitigationBlockDetector
from imbalance_detector import ImbalanceDetector
from liquidity_detector import LiquidityDetector

from analysis_detectors import (
    OrderBlockDetector,
    FairValueGapDetector,
    FractalDetector,
    VolumeContextBuilder,
    detect_market_structure
)
from killzone_detector import KillzoneDetector
from idm_detector import IDMDetector

from analysis_chains import (
    Chain_1_1,
    Chain_1_2,
    Chain_1_3,
    Chain_1_4,
    Chain_1_5,
    Chain_3_2,
    Signal_1,
    Chain_2_6
)

from position_tracker import PositionTracker
from signal_validator import SignalValidator
from signal_deduplicator import SignalDeduplicator
from analysis_interfaces import ChainSignal, Zone, DetectionResult

# --------------------------
#  НАСТРОЙКИ
# --------------------------
from config import (
       BOT_TOKEN,
       CHAT_ID,
       LOG_CHAT_ID,
       SYMBOLS,
       TIMEFRAMES,
       LOOP_INTERVAL,
       POSITION_CLEANUP_INTERVAL,
       MIN_CONFLUENCE_SCORE,
       PERFORMANCE_DATA_FILE,
       CORRELATION_UPDATE_INTERVAL
   )


# ================================
#   ENHANCED ORCHESTRATOR
# ================================

class EnhancedOrchestrator(Orchestrator):
    """
    Расширенный Orchestrator, который сохраняет detections
    """

    def __init__(self, data_source, detectors: dict, chains: list):
        super().__init__(data_source, detectors, chains)
        self.last_detections = {}  # Сохраняем последние detections
        self.last_candles = {}  # Сохраняем последние candles

    async def analyze_symbol_with_data(self, symbol: str):
        """
        Анализирует символ и возвращает signals + detections + candles
        С ИНТЕГРИРОВАННЫМ BIAS DETECTOR
        """
        timeframes = ["1d", "4h", "1h", "15m"]

        await self._log(f"🔍 Analyzing {symbol}", "INFO")

        # --------------------------------------------------------
        # LOAD CANDLES
        # --------------------------------------------------------
        candles = {}
        failed_tfs = []

        for tf in timeframes:
            try:
                data = await self.data_source.get_ohlcv(symbol, tf, limit=300)
                if data is None:
                    data = []
                candles[tf] = data

                if len(data) == 0:
                    failed_tfs.append(tf)

            except Exception as e:
                await self._log(f"❌ Failed {symbol} {tf}: {e}", "ERROR")
                candles[tf] = []

        if failed_tfs:
            await self._log(f"⚠️ {symbol}: No data for {', '.join(failed_tfs)}", "WARNING")

        # --------------------------------------------------------
        # RUN DETECTORS
        # --------------------------------------------------------
        detections = {}
        total_zones = 0

        for tf in timeframes:
            if tf not in candles or len(candles[tf]) == 0:
                detections[tf] = DetectionResult([], None)
                continue

            det_results = []

            for name, detector in self.detectors.items():
                try:
                    if candles[tf] is None or len(candles[tf]) == 0:
                        continue

                    result = detector.detect(candles[tf], tf)
                    if isinstance(result, DetectionResult):
                        det_results.append(result)

                except Exception as e:
                    await self._log(f"❌ Detector {name} failed on {symbol}: {e}", "ERROR")

            # Merge results
            merged_zones = []
            merged_context = None

            for r in det_results:
                if r.zones:
                    merged_zones.extend(r.zones)
                if r.context is not None:
                    merged_context = r.context

            detections[tf] = DetectionResult(merged_zones, merged_context)
            total_zones += len(merged_zones)

        if total_zones > 0:
            await self._log(f"✓ {symbol}: Found {total_zones} zones total", "INFO")

        # --------------------------------------------------------
        # НОВОЕ: АНАЛИЗ BIAS ДЛЯ КАЖДОГО ТАЙМФРЕЙМА
        # --------------------------------------------------------

        bias_detector = ICTBiasDetector()
        bias_contexts = {}

        # Mapping для старших таймфреймов
        htf_map = {
            "15m": "1h",
            "1h": "4h",
            "4h": "1d",
            "1d": None
        }

        for tf in timeframes:
            if tf not in candles or len(candles[tf]) < 50:
                continue

            try:
                # Получаем старший таймфрейм
                htf = htf_map.get(tf)
                htf_candles = candles.get(htf) if htf else None

                # Анализируем bias
                bias_context = bias_detector.detect_comprehensive_bias(
                    candles_current=candles[tf],
                    candles_htf=htf_candles,
                    zones_current=detections[tf].zones if tf in detections else None,
                    tf_current=tf,
                    tf_htf=htf if htf else "4h"
                )

                bias_contexts[tf] = bias_context

                # Логируем bias для отладки
                await self._log(
                    f"📊 {symbol} {tf} BIAS: {bias_context.bias} "
                    f"(strength: {bias_context.strength:.0f}, "
                    f"P/D: {bias_context.premium_discount})",
                    "DEBUG"
                )

            except Exception as e:
                await self._log(f"⚠️ Bias detector failed for {symbol} {tf}: {e}", "DEBUG")
                # Создаем нейтральный bias как fallback
                from bias_detector import BiasContext, MarketBias
                bias_contexts[tf] = BiasContext(
                    bias=MarketBias.NEUTRAL,
                    htf_bias=None,
                    structure_break=None,
                    premium_discount="EQUILIBRIUM",
                    order_flow="NEUTRAL_OF",
                    strength=50.0,
                    key_levels={},
                    notes=["Bias detection failed"]
                )

        # --------------------------------------------------------
        # RUN CHAINS С BIAS CONTEXTS
        # --------------------------------------------------------
        from analysis_interfaces import ChainContext

        ctx = ChainContext(
            symbol=symbol,
            candles=candles,
            detections=detections,
            bias_contexts=bias_contexts,
            # ДОБАВЛЕНО: передаем bias contexts
            log_callback=None if not self.verbose_logging else self.log_callback,
        )

        all_signals = []

        for chain in self.chains:
            try:
                res = await chain.analyze(ctx)
                if res:
                    all_signals.extend(res)

            except Exception as e:
                await self._log(f"❌ Chain {chain.chain_id} failed: {e}", "ERROR")

        if all_signals:
            # Добавляем информацию о bias в описание сигналов
            for sig in all_signals:
                if sig.tf in bias_contexts:
                    bias = bias_contexts[sig.tf]
                    sig.description += f" | Bias: {bias.bias} ({bias.strength:.0f})"

            await self._log(
                f"🎯 {symbol}: {len(all_signals)} signals " +
                f"({', '.join([s.chain_id for s in all_signals])})",
                "INFO"
            )

        # Сохраняем для последующего использования
        self.last_detections[symbol] = detections
        self.last_candles[symbol] = candles
        self.last_bias_contexts = bias_contexts  # НОВОЕ: сохраняем bias

        return all_signals, detections, candles


# ================================
#   CONFLUENCE ANALYZER
# ================================

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
            symbol: str,
            detections: Dict[str, DetectionResult],
            candles: Dict[str, List]
    ) -> List[ConfluenceScore]:

        confluence_zones = []

        # Проверяем каждую зону на младшем TF
        zones_15m = detections.get("15m", DetectionResult([], None)).zones
        if not zones_15m:
            return []

        for zone_15m in zones_15m:
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
            if "15m" in candles and candles["15m"]:
                current_price = candles["15m"][-1].close

                # Зона у round number
                if self._near_round_number(zone_15m):
                    score += 15
                    factors.append("Round number")

                # Зона у дневного high/low
                if "1d" in candles and candles["1d"]:
                    daily_high = max(c.high for c in candles["1d"][-1:])
                    daily_low = min(c.low for c in candles["1d"][-1:])

                    if abs(zone_15m.high - daily_high) / daily_high < 0.001:
                        score += 20
                        factors.append("Daily high")
                    elif abs(zone_15m.low - daily_low) / daily_low < 0.001:
                        score += 20
                        factors.append("Daily low")

            if score >= MIN_CONFLUENCE_SCORE:
                confluence_zones.append(
                    ConfluenceScore(
                        symbol=symbol,
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
            # Для крипты круглые числа
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


# ================================
#   PERFORMANCE OPTIMIZER
# ================================

class PerformanceOptimizer:
    """
    Отслеживает эффективность каждой цепочки и оптимизирует параметры
    """

    def __init__(self, data_file: str = PERFORMANCE_DATA_FILE):
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
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")

    def update_signal_result(
            self,
            chain_id: str,
            symbol: str,
            tf: str,
            outcome: str,  # "TP", "SL", "BE"
            rr_achieved: float,
            entry_time: datetime = None
    ):
        """
        Обновляет статистику после закрытия позиции
        """
        if entry_time is None:
            entry_time = datetime.now()

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
            "day_of_week": entry_time.weekday()
        })

        # Ограничиваем историю последними 500 сигналами
        if len(self.performance_data[key]["signals"]) > 500:
            self.performance_data[key]["signals"] = self.performance_data[key]["signals"][-500:]

        # Пересчитываем статистику
        self._recalculate_stats(key)
        self._save_data()

    def _recalculate_stats(self, key: str):
        """
        Пересчитывает статистику для цепочки
        """
        signals = self.performance_data[key]["signals"]

        if len(signals) < 10:
            return

        # Win rate
        wins = sum(1 for s in signals if s["outcome"] == "TP")
        win_rate = wins / len(signals) if signals else 0

        # Средний RR
        rr_values = [s["rr_achieved"] for s in signals if s["rr_achieved"] is not None]
        avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0

        # Лучший таймфрейм
        tf_performance = {}
        for s in signals:
            tf = s["tf"]
            if tf not in tf_performance:
                tf_performance[tf] = {"wins": 0, "total": 0}
            tf_performance[tf]["total"] += 1
            if s["outcome"] == "TP":
                tf_performance[tf]["wins"] += 1

        best_tf = "15m"  # default
        if tf_performance:
            best_tf = max(
                tf_performance.items(),
                key=lambda x: x[1]["wins"] / x[1]["total"] if x[1]["total"] > 5 else 0
            )[0]

        # Лучшее время суток
        hour_performance = {}
        for s in signals:
            hour = s.get("hour", 0)
            if hour not in hour_performance:
                hour_performance[hour] = {"wins": 0, "total": 0}
            hour_performance[hour]["total"] += 1
            if s["outcome"] == "TP":
                hour_performance[hour]["wins"] += 1

        best_hours = []
        if hour_performance:
            best_hours = sorted(
                hour_performance.items(),
                key=lambda x: x[1]["wins"] / x[1]["total"] if x[1]["total"] > 3 else 0,
                reverse=True
            )[:3]  # Top 3 часа
            best_hours = [h[0] for h in best_hours]

        self.performance_data[key]["stats"] = {
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "best_tf": best_tf,
            "best_hours": best_hours,
            "total_signals": len(signals)
        }

    def should_take_signal(
            self,
            chain_id: str,
            symbol: str,
            tf: str,
            current_time: datetime = None
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
        if stats["win_rate"] < 0.30:  # Меньше 30% - отключаем
            return False, f"Low win rate: {stats['win_rate']:.1%}"

        # Проверяем средний RR
        if stats.get("avg_rr",
                     0) < -0.5:  # Средний убыток больше 0.5R
            return False, f"Negative avg RR: {stats['avg_rr']:.2f}"

        return True, "OK"

    def get_chain_ranking(self) -> List[Tuple[str, float]]:
        """
        Возвращает рейтинг цепочек по эффективности
        """
        rankings = []

        for key, data in self.performance_data.items():
            stats = data.get("stats", {})
            if stats and stats.get("total_signals", 0) >= 20:
                # Формула эффективности: win_rate * avg_rr
                score = stats["win_rate"] * max(stats.get("avg_rr", 0), 0)
                chain_id = key.split("_")[0]
                rankings.append((chain_id, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)


# ================================
#   CORRELATION ANALYZER
# ================================

class CorrelationAnalyzer:
    """
    Отслеживает корреляцию между символами
    """

    def __init__(self):
        self.correlation_matrix = {}
        self.last_update = None

    def calculate_correlations(
            self,
            symbols: List[str],
            data_source
    ) -> None:
        """
        Рассчитывает матрицу корреляций (упрощенная версия)
        """
        # Хардкод известных корреляций криптовалют
        known_correlations = {
            "BTC/USDT_ETH/USDT": 0.85,
            "ETH/USDT_BTC/USDT": 0.85,
            "BNB/USDT_ETH/USDT": 0.75,
            "ETH/USDT_BNB/USDT": 0.75,
            "SOL/USDT_AVAX/USDT": 0.70,
            "AVAX/USDT_SOL/USDT": 0.70,
            "DOT/USDT_ATOM/USDT": 0.65,
            "ATOM/USDT_DOT/USDT": 0.65,
        }

        self.correlation_matrix = known_correlations
        self.last_update = datetime.now()

    def check_correlation_conflict(
            self,
            new_signal: ChainSignal,
            active_signals: List[ChainSignal]
    ) -> Tuple[bool, str]:
        """
        Проверяет конфликт с активными сигналами на коррелирующих парах
        """
        for active in active_signals:
            if active.symbol == new_signal.symbol:
                continue

            key = f"{new_signal.symbol}_{active.symbol}"
            corr = self.correlation_matrix.get(key, 0)

            # Высокая положительная корреляция
            if corr > 0.7:
                # Сигналы должны быть в одном направлении
                if new_signal.direction != active.direction:
                    return True, f"Correlation conflict with {active.symbol}"

            # Высокая отрицательная корреляция
            elif corr < -0.7:
                # Сигналы должны быть в противоположных направлениях
                if new_signal.direction == active.direction:
                    return True, f"Inverse correlation conflict with {active.symbol}"

        return False, "OK"


# ================================
#   ADVANCED SIGNAL FILTER
# ================================

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
        УЛУЧШЕННАЯ проверка возраста зон с учетом таймфрейма
        """
        if not candles:
            return True, "OK"

        current_candle_index = len(candles)

        # ИСПРАВЛЕНО: Разные лимиты для разных таймфреймов
        # Определяем максимальный возраст зоны в зависимости от TF
        tf_age_limits = {
            "15m": 200,  # 200 свечей = ~50 часов
            "30m": 150,  # 150 свечей = ~75 часов
            "1h": 120,  # 120 свечей = 5 дней
            "4h": 100,  # 100 свечей = ~16 дней
            "1d": 50,  # 50 свечей = 50 дней
        }

        # Берем лимит для текущего TF сигнала
        max_age = tf_age_limits.get(signal.tf, 100)

        for zone in zones:
            if hasattr(zone, 'candle_index') and zone.candle_index:
                age = current_candle_index - zone.candle_index

                # НОВОЕ: Учитываем таймфрейм самой зоны
                if hasattr(zone, 'tf') and zone.tf:
                    # Если зона со старшего TF, увеличиваем допустимый возраст
                    zone_tf_limit = tf_age_limits.get(zone.tf, 100)

                    # Используем больший из двух лимитов
                    effective_limit = max(max_age, zone_tf_limit)

                    # Для зон старших TF применяем множитель
                    if zone.tf in ["4h", "1d"]:
                        effective_limit = effective_limit * 1.5  # +50% для старших TF

                    if age > effective_limit:
                        return False, f"Zone too old ({age} candles > {effective_limit} limit for {zone.tf})"
                else:
                    # Fallback на оригинальную логику
                    if age > max_age:
                        return False, f"Zone too old ({age} candles > {max_age} limit)"

        return True, "OK"

    def filter_by_momentum(
            self,
            signal: ChainSignal,
            candles: List
    ) -> Tuple[bool, str]:
        """
        Проверяет импульс движения (простой RSI)
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


# ================================
#   HEARTBEAT
# ================================

async def heartbeat(log):
    while True:
        await log("💓 Heartbeat – bot alive", to_telegram=False)
        await asyncio.sleep(300)  # каждые 5 минут


# ================================
#   ГЛАВНАЯ ПРОГРАММА
# ================================

async def main():
    print("🚀 Starting Smart Money bot (FIXED VERSION)")

    # Источник данных
    source = BinanceDataSource()

    # Паблишеры
    signal_publisher = TelegramSignalPublisher(BOT_TOKEN, CHAT_ID)
    log_publisher = TelegramLogPublisher(BOT_TOKEN, LOG_CHAT_ID)

    # Инициализируем закреплённое сообщение со статистикой
    await signal_publisher.init_pinned_message()

    # Трекер позиций
    position_tracker = PositionTracker(publisher=signal_publisher)
    position_tracker.debug_mode = False

    # Валидатор сигналов
    validator = SignalValidator()

    # НОВЫЕ КОМПОНЕНТЫ
    confluence_analyzer = ConfluenceAnalyzer()
    performance_optimizer = PerformanceOptimizer()
    correlation_analyzer = CorrelationAnalyzer()
    advanced_filter = AdvancedSignalFilter()

    # Функция логирования
    async def log(msg: str, to_telegram: bool = False):
        print(msg)
        if to_telegram:
            try:
                await log_publisher.send_log(msg)
            except:
                pass

    # Запуск heartbeat
    asyncio.create_task(heartbeat(log))

    # Детекторы
    detectors = {
        "OrderBlock": OrderBlockDetector(),
        "FairValueGap": FairValueGapDetector(),
        "Fractal": FractalDetector(),
        "VolumeContext": VolumeContextBuilder(),
        "IDM": IDMDetector(),
    }

    # Цепочки
    chains = [
        Chain_1_1(),
        Chain_1_2(),
        Chain_1_3(),
        Chain_1_4(),
        Chain_1_5(),
        Chain_3_2(),
        Signal_1(),
        Chain_2_6()
    ]

    # ИСПОЛЬЗУЕМ РАСШИРЕННЫЙ ORCHESTRATOR
    orchestrator = EnhancedOrchestrator(source, detectors, chains)
    orchestrator.set_logger(log, verbose=False)

    # Кеши и счетчики
    deduplicator = SignalDeduplicator()  # Умная дедупликация
    active_signals_per_symbol = {}
    cycle_count = 0

    # ЗАЩИТА ОТ ДУБЛИРОВАНИЯ СИГНАЛОВ
    last_signal_time = {}  # {symbol-direction: timestamp} для контроля частоты
    MIN_SIGNAL_INTERVAL = 300000  # 5 минут между сигналами в одном направлении
    MAX_SIGNALS_PER_SYMBOL = 2  # Максимум 2 активных сигнала на символ
    signal_count_per_symbol = {}  # {symbol: count} счетчик активных сигналов

    # Для отслеживания активных сигналов (для проверки корреляций)
    all_active_signals = []

    try:
        while True:
            start = time.time()
            cycle_count += 1

            print("\n" + "=" * 50)
            print(f"🔎 SCAN CYCLE #{cycle_count} STARTED")
            print("=" * 50)

            # Обновляем корреляции периодически
            if cycle_count % CORRELATION_UPDATE_INTERVAL == 1:
                print("📊 Updating correlations...")
                correlation_analyzer.calculate_correlations(SYMBOLS, source)

            # Отправляем в Telegram только важное
            if cycle_count % 10 == 1:
                await log(f"🔎 Cycle #{cycle_count} started", to_telegram=True)

            # Счетчики для статистики
            total_signals = 0
            validated_signals = 0
            confluence_filtered = 0
            performance_filtered = 0
            signals_by_chain = {}

            for index, symbol in enumerate(SYMBOLS):
                print(f"\n[{index + 1}/{len(SYMBOLS)}] Scanning {symbol}...")
                t0 = time.time()

                # ---------------------------------------
                # 1) АНАЛИЗ СИГНАЛОВ С ПОЛУЧЕНИЕМ ДАННЫХ
                # ---------------------------------------
                try:
                    # ИСПОЛЬЗУЕМ НОВЫЙ МЕТОД
                    signals, detections, candles_dict = await orchestrator.analyze_symbol_with_data(symbol)

                    if signals:
                        print(f"  → Found {len(signals)} raw signals")
                        total_signals += len(signals)

                        # Получаем контексты для валидации
                        contexts = {}
                        for tf in TIMEFRAMES:
                            if tf in candles_dict and candles_dict[tf] and len(candles_dict[tf]) > 10:
                                contexts[tf] = detect_market_structure(candles_dict[tf], tf)

                        # ---------------------------------------
                        # 2) ВАЛИДАЦИЯ СИГНАЛОВ
                        # ---------------------------------------
                        valid_signals = validator.filter_signals(signals, contexts)
                        validated_signals += len(valid_signals)

                        if valid_signals:
                            print(f"  ✔ {len(valid_signals)} signals passed validation")

                            # Считаем по цепочкам
                            for sig in valid_signals:
                                if sig.chain_id not in signals_by_chain:
                                    signals_by_chain[sig.chain_id] = 0
                                signals_by_chain[sig.chain_id] += 1
                        else:
                            print(f"  âš  All signals filtered out")

                        signals = valid_signals

                        # ---------------------------------------
                        # 3) АНАЛИЗ КОНФЛЮЕНЦИИ (ТЕПЕРЬ РАБОТАЕТ)
                        # ---------------------------------------
                        if signals and detections:  # Теперь у нас есть detections!
                            print(f"  🔍 Analyzing confluence...")
                            confluence_scores = confluence_analyzer.analyze_confluence(
                                symbol=symbol,
                                detections=detections,
                                candles=candles_dict
                            )

                            # Фильтруем по конфлюенции
                            if confluence_scores:
                                print(f"    Found {len(confluence_scores)} confluence zones")
                                high_confluence_zones = [c.zone for c in confluence_scores if
                                                         c.score >= MIN_CONFLUENCE_SCORE]

                                if high_confluence_zones:
                                    print(
                                        f"    {len(high_confluence_zones)} zones with score >= {MIN_CONFLUENCE_SCORE}")

                                filtered_by_confluence = []
                                for sig in signals:
                                    # Проверяем, есть ли сигнал в зоне с высокой конфлюенцией
                                    sig_in_confluence = False
                                    for zone in high_confluence_zones:
                                        if zone.low <= sig.entry <= zone.high:
                                            sig_in_confluence = True
                                            break

                                    if sig_in_confluence:
                                        filtered_by_confluence.append(sig)
                                    else:
                                        confluence_filtered += 1
                                        print(f"    ✗ {sig.chain_id} - low confluence")

                                signals = filtered_by_confluence

                        # ---------------------------------------
                        # 4) ПРОВЕРКА ПРОИЗВОДИТЕЛЬНОСТИ
                        # ---------------------------------------
                        if signals:
                            performance_checked = []
                            for sig in signals:
                                should_take, reason = performance_optimizer.should_take_signal(
                                    chain_id=sig.chain_id,
                                    symbol=sig.symbol,
                                    tf=sig.tf,
                                    current_time=datetime.now()
                                )

                                if should_take:
                                    performance_checked.append(sig)
                                else:
                                    performance_filtered += 1
                                    print(f"    ✗ {sig.chain_id} - {reason}")

                            signals = performance_checked

                        # ---------------------------------------
                        # 5) РАСШИРЕННАЯ ФИЛЬТРАЦИЯ
                        # ---------------------------------------
                        if signals and candles_dict.get(sig.tf):
                            advanced_filtered = []
                            for sig in signals:
                                # Получаем зоны для текущего TF
                                current_zones = detections.get(sig.tf, DetectionResult([], None)).zones

                                # Проверка возраста зон
                                passed, reason = advanced_filter.filter_by_zone_age(
                                    sig,
                                    current_zones,
                                    candles_dict.get(sig.tf, [])
                                )
                                if not passed:
                                    print(f"    ✗ {sig.chain_id} - {reason}")
                                    continue

                                # Проверка momentum
                                passed, reason = advanced_filter.filter_by_momentum(
                                    sig,
                                    candles_dict.get(sig.tf, [])
                                )
                                if not passed:
                                    print(f"    ✗ {sig.chain_id} - {reason}")
                                    continue

                                # Проверка волатильности
                                passed, reason = advanced_filter.filter_by_volatility(
                                    sig,
                                    candles_dict.get(sig.tf, [])
                                )
                                if not passed:
                                    print(f"    ✗ {sig.chain_id} - {reason}")
                                    continue

                                advanced_filtered.append(sig)

                            signals = advanced_filtered

                        # ---------------------------------------
                        # 6) ПРОВЕРКА КОРРЕЛЯЦИЙ
                        # ---------------------------------------
                        if signals and all_active_signals:
                            correlation_filtered = []
                            for sig in signals:
                                has_conflict, reason = correlation_analyzer.check_correlation_conflict(
                                    sig, all_active_signals
                                )

                                if not has_conflict:
                                    correlation_filtered.append(sig)
                                else:
                                    print(f"    ✗ {sig.chain_id} - {reason}")

                            signals = correlation_filtered

                    else:
                        print(f"  ∅ No signals found")
                        signals = []

                except Exception as e:
                    print(f"  ❌ ERROR: {e}")
                    import traceback
                    print(traceback.format_exc())
                    await log(f"❌ Error analyzing {symbol}: {e}", to_telegram=True)
                    signals = []
                    continue

                t1 = time.time()
                print(f"  ⏱ Time: {t1 - t0:.2f} sec")

                now_ms = int(time.time() * 1000)

                # ---------------------------------------
                # 7) ПРОВЕРКА НА КОНФЛИКТЫ С АКТИВНЫМИ ПОЗИЦИЯМИ
                # ---------------------------------------
                if symbol in active_signals_per_symbol and signals:
                    active_directions = active_signals_per_symbol[symbol]
                    filtered_signals = []

                    for sig in signals:
                        sig_dir = str(sig.direction).upper().replace("DIRECTION.", "")

                        conflict = False
                        for active_dir in active_directions:
                            if (sig_dir in ["LONG", "BUY"] and active_dir in ["SHORT", "SELL"]) or \
                                    (sig_dir in ["SHORT", "SELL"] and active_dir in ["LONG", "BUY"]):
                                print(
                                    f"  âš  Skipping {sig.chain_id} {sig_dir} - active {active_dir} position exists")
                                conflict = True
                                break

                        if not conflict:
                            filtered_signals.append(sig)

                    signals = filtered_signals

                # ---------------------------------------
                # 8) РЕГИСТРАЦИЯ НОВЫХ СИГНАЛОВ
                # ---------------------------------------
                for s in signals:
                    # Проверяем дубликат через умный дедупликатор
                    is_dup, reason = deduplicator.is_duplicate(s, tf=s.tf)

                    if is_dup:
                        print(f"  🔁 Duplicate: {reason}")
                        continue

                    print(f"  📨 Sending: {s.chain_id} {s.direction}")
                    try:
                        await signal_publisher.publish(s)
                        deduplicator.register_signal(s, tf=s.tf)
                        position_tracker.register_signal(s, now_ms)
                        all_active_signals.append(s)

                        if symbol not in active_signals_per_symbol:
                            active_signals_per_symbol[symbol] = set()
                        dir_normalized = str(s.direction).upper().replace("DIRECTION.", "")
                        active_signals_per_symbol[symbol].add(dir_normalized)

                        await log(f"🏁 NEW: {s.symbol} {s.chain_id} {s.direction} RR={s.rr:.1f}", to_telegram=True)
                    except Exception as e:
                        print(f"  âš  Failed: {e}")

                # ---------------------------------------
                # 9) ОБНОВЛЯЕМ СТАТУС ПОЗИЦИЙ
                # ---------------------------------------
                try:
                    candles_15m = await source.get_ohlcv(symbol, "15m", limit=1)
                    if candles_15m:
                        last_candle = candles_15m[-1]
                        await position_tracker.update_with_candle(symbol, last_candle)

                        # Обновляем активные направления на основе открытых позиций
                        active_positions = []
                        for pos in position_tracker.positions.get(symbol, []):
                            if pos.status.value in ["PENDING", "OPEN"]:
                                active_positions.append(pos.direction)

                        if active_positions:
                            active_signals_per_symbol[symbol] = set(active_positions)
                        elif symbol in active_signals_per_symbol:
                            del active_signals_per_symbol[symbol]

                except Exception as e:
                    print(f"  âš  Failed to update positions: {e}")

                # Небольшая задержка между символами
                await asyncio.sleep(0.1)

            # Очищаем старые активные сигналы
            all_active_signals = [s for s in all_active_signals
                                  if s.symbol in active_signals_per_symbol]

            # ---------------------------------------
            # 10) СТАТИСТИКА ЦИКЛА
            # ---------------------------------------
            total_time = time.time() - start

            print("\n" + "=" * 50)
            print(f"📊 CYCLE #{cycle_count} STATISTICS:")
            print(f"• Total time: {total_time:.2f} sec")
            print(f"• Signals found: {total_signals}")
            print(f"• After validation: {validated_signals}")
            print(f"• Confluence filtered: {confluence_filtered}")
            print(f"• Performance filtered: {performance_filtered}")
            print(f"• Unique cached: {deduplicator.get_stats()["active_zones"]}")
            print(f"• Active symbols: {len(active_signals_per_symbol)}")

            if signals_by_chain:
                print("\n🔗 Signals by chain:")
                for chain_id, count in sorted(signals_by_chain.items()):
                    print(f"  • {chain_id}: {count}")

            # Статистика по трекеру
            tracker_stats = position_tracker.get_stats()

            # Обновляем закреплённое сообщение со статистикой
            signal_publisher.update_stats_from_tracker(tracker_stats)

            # Синхронизируем активные позиции для отображения в закрепе
            signal_publisher.active_positions.clear()
            for sym, positions in position_tracker.positions.items():
                for pos in positions:
                    if pos.status.value == "OPEN":
                        signal_publisher.add_active_position(sym, str(pos.direction), pos.entry)

            # Обновляем закреп
            await signal_publisher.update_pinned_stats()

            print(f"\n📈 POSITION TRACKER:")
            print(f"• Total: {tracker_stats['total']}")
            print(f"• Pending: {tracker_stats['pending']}")
            print(f"• Open: {tracker_stats['open']}")
            print(f"• Closed TP: {tracker_stats['closed_tp']}")
            print(f"• Closed SL: {tracker_stats['closed_sl']}")
            print(f"• Cancelled: {tracker_stats['cancelled']}")

            # Win rate если есть закрытые позиции
            total_closed = tracker_stats['closed_tp'] + tracker_stats['closed_sl']
            if total_closed > 0:
                win_rate = (tracker_stats['closed_tp'] / total_closed) * 100
                print(f"• Win Rate: {win_rate:.1f}%")

            # Рейтинг цепочек по производительности
            if cycle_count % 50 == 0:
                rankings = performance_optimizer.get_chain_ranking()
                if rankings:
                    print("\n🏆 Chain Performance Rankings:")
                    for i, (chain_id, score) in enumerate(rankings[:5], 1):
                        print(f"  {i}. {chain_id}: {score:.3f}")

            # Периодическая отправка статистики в Telegram
            if validated_signals > 0 or cycle_count % 20 == 0:
                stats_msg = (
                    f"📊 Cycle #{cycle_count}\n"
                    f"Time: {total_time:.1f}s\n"
                    f"Signals: {validated_signals}\n"
                    f"Filtered: {confluence_filtered + performance_filtered}\n"
                    f"Positions: {tracker_stats['open']} open, {tracker_stats['pending']} pending\n"
                    f"Results: {tracker_stats['closed_tp']}✅ / {tracker_stats['closed_sl']}❌"
                )

                if total_closed > 10:
                    stats_msg += f"\nWin Rate: {win_rate:.1f}%"

                await log(stats_msg, to_telegram=True)

            print("=" * 50)

            # ---------------------------------------
            # 11) ОЧИСТКА СТАРЫХ ДАННЫХ
            # ---------------------------------------
            if cycle_count % 100 == 0 and deduplicator.get_stats()["active_zones"] > 500:
                print("🧹 Clearing old cache...")
                deduplicator.cleanup()

            if cycle_count % POSITION_CLEANUP_INTERVAL == 0:
                removed = position_tracker.cleanup_old_positions()
                if removed > 0:
                    print(f"🧹 Removed {removed} old positions")

            # ---------------------------------------
            # 12) ОЖИДАНИЕ СЛЕДУЮЩЕГО ЦИКЛА
            # ---------------------------------------
            next_time = start + LOOP_INTERVAL
            now = time.time()

            if now < next_time:
                wait = next_time - now
                print(f"\n⏳ Next scan in {int(wait)} seconds...")
                await asyncio.sleep(wait)
            else:
                print("\nâš  Scan took longer than interval, starting immediately...")

    except KeyboardInterrupt:
        print("\n🛑 Stopping bot (KeyboardInterrupt)")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        await log(f"❌ CRITICAL ERROR: {e}", to_telegram=True)
        import traceback
        print(traceback.format_exc())

    finally:
        try:
            await source.close()
            print("✅ Binance connection closed")
        except:
            pass

        # Финальная статистика
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS:")
        final_stats = position_tracker.get_stats()
        print(f"• Total positions: {final_stats['total']}")
        print(f"• Win rate: {final_stats['closed_tp']}/{final_stats['total']}")
        print(f"• Total cycles: {cycle_count}")

        # Статистика по цепочкам
        if final_stats['by_chain']:
            print("\n🔗 Performance by chain:")
            for chain_id, stats in final_stats['by_chain'].items():
                if stats['total'] > 0:
                    wins = stats.get('tp', 0)
                    losses = stats.get('sl', 0)
                    total = wins + losses
                    if total > 0:
                        wr = (wins / total) * 100
                        print(f"  • {chain_id}: {wins}✅/{losses}❌ (WR: {wr:.1f}%)")

        # Финальный рейтинг цепочек
        rankings = performance_optimizer.get_chain_ranking()
        if rankings:
            print("\n🏆 Final Chain Rankings:")
            for i, (chain_id, score) in enumerate(rankings[:5], 1):
                print(f"  {i}. {chain_id}: Score {score:.3f}")

        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())