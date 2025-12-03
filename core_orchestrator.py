import traceback
from analysis_interfaces import ChainContext, DetectionResult
from bias_detector import ICTBiasDetector, BiasContext

class Orchestrator:
    """
    Главный управляющий модуль (с оптимизированным логированием)
    """

    def __init__(self, data_source, detectors: dict, chains: list):
        self.data_source = data_source
        self.detectors = detectors
        self.chains = chains
        self.log_callback = None
        self.verbose_logging = False  # ДОБАВЛЕНО: флаг подробного логирования
        self.log_buffer = []  # ДОБАВЛЕНО: буфер для группировки логов

    def set_logger(self, logger_func, verbose: bool = False):
        """
        Устанавливает logger с опцией verbose.
        verbose=False - только важные события
        verbose=True - детальное логирование
        """
        self.log_callback = logger_func
        self.verbose_logging = verbose

    async def _log(self, msg: str, level: str = "INFO"):
        """
        level: DEBUG, INFO, WARNING, ERROR
        """
        # В режиме без verbose пропускаем DEBUG логи
        if not self.verbose_logging and level == "DEBUG":
            return

        # Всегда печатаем локально (быстро)
        print(msg)

        # В Telegram отправляем только важное
        if self.log_callback and level in ["WARNING", "ERROR"]:
            try:
                await self.log_callback(msg)
            except:
                pass

    def _batch_log(self, msg: str):
        """Добавляет сообщение в буфер для пакетной отправки"""
        self.log_buffer.append(msg)

    async def _flush_logs(self):
        """Отправляет накопленные логи одним сообщением"""
        if not self.log_buffer or not self.log_callback:
            return

        batch_msg = "\n".join(self.log_buffer[:50])  # Максимум 50 строк
        self.log_buffer = self.log_buffer[50:]

        try:
            await self.log_callback(batch_msg)
        except:
            pass

    async def analyze_symbol(self, symbol: str):
        """
        Оптимизированная версия с минимальным логированием
        """
        timeframes = ["1d", "4h", "1h", "15m"]

        # Только важное событие
        await self._log(f"🔍 Analyzing {symbol}", "INFO")

        # --------------------------------------------------------
        # LOAD CANDLES (без детального логирования)
        # --------------------------------------------------------
        candles = {}
        failed_tfs = []

        for tf in timeframes:
            try:
                data = await self.data_source.get_ohlcv(symbol, tf, limit=300)
                if data is None:
                    data = []
                candles[tf] = data

                # DEBUG лог - не отправляется в Telegram
                if len(data) == 0:
                    failed_tfs.append(tf)

            except Exception as e:
                await self._log(f"❌ Failed {symbol} {tf}: {e}", "ERROR")
                candles[tf] = []

        # Одно сообщение вместо множества
        if failed_tfs:
            await self._log(f"⚠️ {symbol}: No data for {', '.join(failed_tfs)}", "WARNING")

        # --------------------------------------------------------
        # RUN DETECTORS (без логирования каждой зоны)
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
                    # Только критические ошибки
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

        # ОДНО сообщение о результатах детекторов
        if total_zones > 0:
            await self._log(f"✓ {symbol}: Found {total_zones} zones total", "INFO")

        # --------------------------------------------------------
        # RUN CHAINS (минимальное логирование)
        # --------------------------------------------------------
        ctx = ChainContext(
            symbol=symbol,
            candles=candles,
            detections=detections,
            log_callback=None if not self.verbose_logging else self.log_callback,
        )

        all_signals = []

        for chain in self.chains:
            try:
                # Цепочки логируют только если verbose=True
                res = await chain.analyze(ctx)
                if res:
                    all_signals.extend(res)

            except Exception as e:
                await self._log(f"❌ Chain {chain.chain_id} failed: {e}", "ERROR")

        # Финальное сообщение
        if all_signals:
            await self._log(
                f"🎯 {symbol}: {len(all_signals)} signals " +
                f"({', '.join([s.chain_id for s in all_signals])})",
                "INFO"
            )

        return all_signals