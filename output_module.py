import aiohttp
from analysis_interfaces import ChainSignal
from typing import Optional, List
from position_tracker import TrackedPosition

# Импорт генератора графиков - приоритет chart-img.com API
try:
    from chart_generator_api import (
        ChartGenerator, 
        ChartGeneratorAPI,
        SignalData, 
        ZoneData, 
        CHARTS_AVAILABLE
    )
    CHART_TYPE = "API"
    print("✅ chart-img.com API charts enabled")
except ImportError:
    try:
        from chart_generator_tv import (
            ChartGenerator, 
            SignalData, 
            ZoneData, 
            CHARTS_AVAILABLE
        )
        CHART_TYPE = "TV"
        print("✅ TradingView Playwright charts enabled (fallback)")
    except ImportError:
        CHARTS_AVAILABLE = False
        ChartGenerator = None
        ChartGeneratorAPI = None
        SignalData = None
        ZoneData = None
        CHART_TYPE = None
        print("⚠️ Charts disabled")


class TelegramSignalPublisher:
    """
    Отправка торговых сигналов в Telegram-канал.
    С русскими объяснениями для СНГ аудитории.
    """

    # === СЛОВАРИ ДЛЯ РУСИФИКАЦИИ ===

    CHAIN_DESCRIPTIONS = {
        "1.1": {
            "name": "Multi-TF Confluence",
            "probability": 75,
            "logic": "Подтверждение на 4 таймфреймах (D→4H→1H→15m)",
        },
        "1.2": {
            "name": "IDM Cascade",
            "probability": 70,
            "logic": "Каскад inducement (сбор стопов) на D и 4H",
        },
        "1.3": {
            "name": "Daily POI + 4H IDM",
            "probability": 68,
            "logic": "Дневная зона интереса + inducement на 4H",
        },
        "1.4": {
            "name": "Daily FVG Fill",
            "probability": 62,
            "logic": "Заполнение дневного имбаланса (FVG)",
        },
        "1.5": {
            "name": "FVG Reaction",
            "probability": 58,
            "logic": "Реакция от дневного FVG на младшем TF",
        },
        "2.6": {
            "name": "Liquidity Sweep",
            "probability": 72,
            "logic": "Снятие ликвидности (sweep) на 4H + вход на 15m FVG",
        },
        "3.2": {
            "name": "First Touch FVG",
            "probability": 65,
            "logic": "Первое касание нетронутого FVG на 4H",
        },
        "Signal_1": {
            "name": "FH + OB Combo",
            "probability": 60,
            "logic": "Fractal High/Low + Order Block на 4H",
        },
    }

    DIRECTION_RU = {
        "LONG": "🟢 ЛОНГ",
        "SHORT": "🔴 ШОРТ",
    }

    def __init__(self, bot_token: str, chat_id: str):
        self.token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.photo_url = f"https://api.telegram.org/bot{self.token}/sendPhoto"

        # Для закреплённого сообщения
        self.pinned_message_id: Optional[int] = None
        self.active_positions: dict = {}
        self.stats = {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "open": 0,
            "partial": 0,
            "total_rr": 0.0,
        }
        
        # Инициализация генератора графиков
        self.chart_generator: Optional[ChartGenerator] = None
        if CHARTS_AVAILABLE:
            try:
                self.chart_generator = ChartGenerator(
                    width=1200,
                    height=800,
                )
                print(f"✅ Chart generator initialized ({CHART_TYPE})")
            except Exception as e:
                print(f"⚠️ Chart generator init failed: {e}")
                self.chart_generator = None

    async def _send(self, text: str, reply_to_message_id: Optional[int] = None) -> Optional[int]:
        """
        Внутренний метод отправки сообщения в Telegram.
        Возвращает message_id отправленного сообщения.
        """
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"❌ Failed to send to Telegram: HTTP {resp.status} → {body}")
                        return None
                    data = await resp.json()
                    return data.get("result", {}).get("message_id")
        except Exception as e:
            print(f"❌ Error sending to Telegram: {e}")
            return None

    async def _send_photo(
        self, 
        image_bytes: bytes, 
        caption: str, 
        reply_to_message_id: Optional[int] = None
    ) -> Optional[int]:
        """
        Отправляет фото в Telegram.
        Возвращает message_id отправленного сообщения.
        """
        form_data = aiohttp.FormData()
        form_data.add_field('chat_id', self.chat_id)
        form_data.add_field('photo', image_bytes, filename='chart.png', content_type='image/png')
        form_data.add_field('caption', caption[:1024])  # Telegram limit
        form_data.add_field('parse_mode', 'HTML')
        
        if reply_to_message_id:
            form_data.add_field('reply_to_message_id', str(reply_to_message_id))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.photo_url, data=form_data) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"❌ Failed to send photo to Telegram: HTTP {resp.status} → {body}")
                        return None
                    data = await resp.json()
                    return data.get("result", {}).get("message_id")
        except Exception as e:
            print(f"❌ Error sending photo to Telegram: {e}")
            return None

    async def publish_position_opened(self, pos: TrackedPosition):
        """
        Сообщение о том, что лимитка активировалась (позиция открыта).
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")
        direction_emoji = "🟢" if "LONG" in direction_str else "🔴"

        text = (
            f"{direction_emoji} <b>ПОЗИЦИЯ ОТКРЫТА</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Направление: <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n"
            f"SL: <b>{pos.stop_loss:.5f}</b>\n"
            f"TPs: <b>{', '.join(f'{x:.5f}' for x in pos.take_profits)}</b>\n"
            f"RR: <b>{pos.rr:.2f}</b>"
        )
        await self._send(text, reply_to_message_id=pos.signal_message_id)

    async def publish_tp1_hit(self, pos: TrackedPosition, tp1_rr: float):
        """
        Сообщение о достижении TP1 с рекомендацией зафиксировать часть и перенести в БУ.
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")
        direction_emoji = "🟢" if "LONG" in direction_str else "🔴"

        text = (
            f"🎯 <b>TP1 ДОСТИГНУТ!</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Направление: {direction_emoji} <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n"
            f"TP1: <b>{pos.take_profits[0]:.5f}</b>\n\n"
            f"📊 <b>Результат: +{tp1_rr:.2f}R</b> (50% позиции)\n\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💡 <b>РЕКОМЕНДАЦИЯ:</b>\n"
            f"• Зафиксируйте 50% позиции\n"
            f"• Стоп перенесён в безубыток ({pos.entry:.5f})\n"
            f"• Ждём TP2: <b>{pos.take_profits[1]:.5f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━"
        )
        await self._send(text, reply_to_message_id=pos.signal_message_id)

    async def publish_position_breakeven(self, pos: TrackedPosition):
        """
        Сообщение о срабатывании безубытка после TP1.
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")

        text = (
            f"⚪ <b>БЕЗУБЫТОК</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Направление: <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n\n"
            f"📊 <b>Итог: +{pos.realized_rr:.2f}R</b>\n"
            f"<i>(TP1 взят на 50%, остаток закрыт в БУ)</i>"
        )
        await self._send(text, reply_to_message_id=pos.signal_message_id)

    async def publish_position_closed(self, pos: TrackedPosition, hit_tp_index: Optional[int]):
        """
        Сообщение о TP/SL с указанием RR.
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")

        # Рассчитываем RR
        original_risk = abs(pos.entry - pos.original_stop_loss) if pos.original_stop_loss else abs(pos.entry - pos.stop_loss)
        
        if pos.outcome == "SL":
            icon = "🔴"
            outcome_text = "<b>Стоп-лосс сработал</b>"
            rr_text = f"📊 <b>Результат: -1.00R</b>"
        elif pos.outcome == "TP2":
            icon = "🟢"
            outcome_text = "<b>Take Profit 2 достигнут!</b>"
            rr_text = f"📊 <b>Результат: +{pos.realized_rr:.2f}R</b>"
        elif pos.outcome == "BE":
            icon = "⚪"
            outcome_text = "<b>Безубыток после TP1</b>"
            rr_text = f"📊 <b>Результат: +{pos.realized_rr:.2f}R</b>"
        else:
            icon = "🟢"
            if hit_tp_index is not None:
                tp_price = pos.take_profits[hit_tp_index] if hit_tp_index < len(pos.take_profits) else pos.entry
                tp_distance = abs(tp_price - pos.entry)
                achieved_rr = tp_distance / original_risk if original_risk > 0 else 0
                outcome_text = f"<b>Take Profit {hit_tp_index + 1} достигнут!</b>"
                rr_text = f"📊 <b>Результат: +{achieved_rr:.2f}R</b>"
            else:
                outcome_text = "<b>Позиция закрыта</b>"
                rr_text = f"📊 <b>Результат: +{pos.realized_rr:.2f}R</b>"

        text = (
            f"{icon} <b>ПОЗИЦИЯ ЗАКРЫТА</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Результат: {outcome_text}\n"
            f"Направление: <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n"
            f"SL: <b>{pos.original_stop_loss:.5f}</b>\n"
            f"TPs: <b>{', '.join(f'{x:.5f}' for x in pos.take_profits)}</b>\n\n"
            f"{rr_text}"
        )
        await self._send(text, reply_to_message_id=pos.signal_message_id)

    async def publish_position_cancelled(self, pos: TrackedPosition, reason: str):
        """
        Сообщение о том, что лимитка отменена.
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")

        text = (
            f"⚪ <b>ОРДЕР ОТМЕНЁН</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Причина: <i>{reason}</i>\n"
            f"Направление: <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n"
            f"SL: <b>{pos.stop_loss:.5f}</b>\n"
            f"TPs: <b>{', '.join(f'{x:.5f}' for x in pos.take_profits)}</b>"
        )
        await self._send(text, reply_to_message_id=pos.signal_message_id)

    def _fmt_price(self, p: float) -> str:
        """Форматирует цену в зависимости от величины"""
        if p >= 1000:
            return f"{p:,.2f}"
        elif p >= 1:
            return f"{p:.4f}"
        else:
            return f"{p:.6f}"

    async def publish(self, signal: ChainSignal) -> Optional[int]:
        """
        Отправка нового сигнала в Telegram с русскими объяснениями.
        Возвращает message_id для последующих reply.
        """
        # Чистим direction
        direction_str = str(signal.direction).replace("Direction.", "").replace("DIRECTION.", "").upper()
        direction_ru = self.DIRECTION_RU.get(direction_str,
                                             f"{'🟢' if 'LONG' in direction_str else '🔴'} {direction_str}")

        # Получаем инфо о цепочке
        chain_info = self.CHAIN_DESCRIPTIONS.get(signal.chain_id, {
            "name": signal.chain_id,
            "probability": 55,
            "logic": "Multi-TF анализ",
        })

        # Расчёт риска
        risk = abs(signal.entry - signal.stop_loss)
        risk_percent = (risk / signal.entry) * 100

        # TP линии с RR
        tp_lines = ""
        final_rr = 0
        for i, tp in enumerate(signal.take_profits, start=1):
            reward = abs(tp - signal.entry)
            tp_rr = reward / risk if risk > 0 else 0
            tp_percent = (reward / signal.entry) * 100
            tp_lines += f"  TP{i}: <b>{self._fmt_price(tp)}</b> (RR {tp_rr:.1f}x, +{tp_percent:.1f}%)\n"
            final_rr = tp_rr

        # Emoji для направления
        dir_emoji = "🟢" if "LONG" in direction_str else "🔴"

        # Формируем сообщение
        text = f"""<b>{signal.symbol}</b> | {signal.tf.upper()}
{direction_ru}

━━━━━━━━━━━━━━━━━━
<b>СЕТАП:</b> {chain_info['name']}
📊 Вероятность: <b>{chain_info['probability']}%</b>

<i>{chain_info['logic']}</i>
━━━━━━━━━━━━━━━━━━

📍 <b>Entry:</b> {self._fmt_price(signal.entry)}
🛑 <b>Stop:</b> {self._fmt_price(signal.stop_loss)} <i>(-{risk_percent:.1f}%)</i>

🎯 <b>Цели:</b>
{tp_lines}
⚖️ <b>R:R:</b> {final_rr:.1f}x
━━━━━━━━━━━━━━━━━━

⚠️ <i>1-2% риск на сделку.</i>"""

        message_id = await self._send(text)

        # Увеличиваем счётчик сигналов
        self.stats["total_signals"] += 1

        # Обновляем закреп после нового сигнала
        if self.pinned_message_id:
            await self.update_pinned_stats()

        return message_id

    async def publish_with_chart(
        self, 
        signal: ChainSignal, 
        candles: List = None,
        zones: List = None
    ) -> Optional[int]:
        """
        Отправка сигнала С ГРАФИКОМ в Telegram.
        
        Для chart-img.com API candles не нужны - данные берутся с TradingView.
        
        Args:
            signal: Объект сигнала
            candles: Игнорируется для API версии
            zones: Список зон для отрисовки
            
        Returns:
            message_id для последующих reply
        """
        # Если график недоступен - отправляем обычное сообщение
        if not self.chart_generator:
            print("⚠️ Chart not available, sending text only")
            return await self.publish(signal)

        try:
            # Получаем инфо о цепочке
            chain_info = self.CHAIN_DESCRIPTIONS.get(signal.chain_id, {
                "name": signal.chain_id,
                "probability": 55,
                "logic": "Multi-TF анализ",
            })

            # Конвертируем зоны
            chart_zones = []
            if zones and ZoneData:
                for z in zones[:4]:  # Максимум 4 зоны
                    chart_zones.append(ZoneData(
                        low=z.low,
                        high=z.high,
                        zone_type=z.type,
                    ))

            # Создаём данные для графика
            signal_data = SignalData(
                symbol=signal.symbol,
                tf=signal.tf,
                direction=str(signal.direction).replace("Direction.", "").replace("DIRECTION.", "").upper(),
                entry=float(signal.entry),
                stop_loss=float(signal.stop_loss),
                take_profits=[float(tp) for tp in signal.take_profits],
                zones=chart_zones,
                chain_name=chain_info['name'],
                rr=float(signal.rr)
            )

            # Генерируем график
            # API версия - асинхронная, не требует candles
            if CHART_TYPE == "API":
                image_bytes = await self.chart_generator.api_generator.generate(signal_data)
            elif CHART_TYPE == "TV" and hasattr(self.chart_generator, 'tv_generator'):
                # TradingView Playwright версия
                image_bytes = await self.chart_generator.tv_generator.generate(candles, signal_data)
            else:
                # Fallback sync версия
                image_bytes = self.chart_generator.generate(candles, signal_data)
            
            if not image_bytes:
                print("⚠️ Chart generation failed, sending text only")
                return await self.publish(signal)

            # Формируем подпись для фото (сокращённая версия)
            direction_str = str(signal.direction).replace("Direction.", "").replace("DIRECTION.", "").upper()
            direction_ru = self.DIRECTION_RU.get(direction_str,
                                                 f"{'🟢' if 'LONG' in direction_str else '🔴'} {direction_str}")

            risk = abs(signal.entry - signal.stop_loss)
            risk_percent = (risk / signal.entry) * 100

            # TP с RR
            tp_lines = ""
            final_rr = 0
            for i, tp in enumerate(signal.take_profits, start=1):
                reward = abs(tp - signal.entry)
                tp_rr = reward / risk if risk > 0 else 0
                tp_lines += f"TP{i}: {self._fmt_price(tp)} ({tp_rr:.1f}R)\n"
                final_rr = tp_rr

            # Компактный caption (до 1024 символов)
            caption = f"""<b>{signal.symbol}</b> | {signal.tf.upper()}
{direction_ru}

<b>СЕТАП:</b> {chain_info['name']}
📊 Вероятность: <b>{chain_info['probability']}%</b>
<i>{chain_info['logic']}</i>

📍 Entry: <b>{self._fmt_price(signal.entry)}</b>
🛑 Stop: <b>{self._fmt_price(signal.stop_loss)}</b> (-{risk_percent:.1f}%)
🎯 {tp_lines}
⚖️ R:R: <b>{final_rr:.1f}x</b>

⚠️ <i>1-2% риск на сделку</i>"""

            # Отправляем фото
            message_id = await self._send_photo(image_bytes, caption)
            
            if not message_id:
                # Fallback на текст
                print("⚠️ Photo send failed, trying text")
                return await self.publish(signal)

            # Увеличиваем счётчик сигналов
            self.stats["total_signals"] += 1

            # Обновляем закреп
            if self.pinned_message_id:
                await self.update_pinned_stats()

            print(f"✅ Signal with chart sent: {signal.symbol} {signal.chain_id}")
            return message_id

        except Exception as e:
            print(f"❌ publish_with_chart error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback на обычный publish
            return await self.publish(signal)

    async def _edit_message(self, message_id: int, text: str) -> bool:
        """Редактирует существующее сообщение"""
        url = f"https://api.telegram.org/bot{self.token}/editMessageText"
        payload = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except:
            return False

    async def _pin_message(self, message_id: int) -> bool:
        """Закрепляет сообщение"""
        url = f"https://api.telegram.org/bot{self.token}/pinChatMessage"
        payload = {
            "chat_id": self.chat_id,
            "message_id": message_id,
            "disable_notification": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except:
            return False

    async def _send_and_get_id(self, text: str) -> Optional[int]:
        """Отправляет сообщение и возвращает его ID"""
        return await self._send(text)

    async def init_pinned_message(self):
        """Создаёт и закрепляет сообщение со статистикой"""
        text = self._build_stats_message()
        message_id = await self._send_and_get_id(text)
        if message_id:
            self.pinned_message_id = message_id
            await self._pin_message(message_id)
            print(f"📌 Pinned stats message: {message_id}")

    async def update_pinned_stats(self):
        """Обновляет закреплённое сообщение"""
        if not self.pinned_message_id:
            return
        text = self._build_stats_message()
        await self._edit_message(self.pinned_message_id, text)

    def _build_stats_message(self) -> str:
        """Формирует текст статистики с RR"""
        from datetime import datetime
        now = datetime.now()

        total_closed = self.stats["wins"] + self.stats["losses"]
        win_rate = f"{(self.stats['wins'] / total_closed * 100):.1f}%" if total_closed > 0 else "—"
        
        # Форматируем RR
        total_rr = self.stats.get("total_rr", 0.0)
        rr_emoji = "📈" if total_rr >= 0 else "📉"
        rr_sign = "+" if total_rr >= 0 else ""

        # Список активных позиций
        pos_lines = []
        for key, pos in list(self.active_positions.items())[:10]:
            emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
            status = "🎯" if pos.get("partial") else ""
            pos_lines.append(f"  {emoji} <b>{pos['symbol']}</b> @ {pos['entry']:.2f} {status}")

        positions_text = "\n".join(pos_lines) if pos_lines else "  <i>Нет активных</i>"

        return f"""📊 <b>ICT/SMC BOT STATUS</b>
━━━━━━━━━━━━━━━━━━

🕐 Обновлено: {now.strftime("%H:%M:%S")}

📈 <b>СТАТИСТИКА</b>
  • Всего сигналов: {self.stats['total_signals']}
  • Win Rate: <b>{win_rate}</b>
  • Wins: {self.stats['wins']} ✅
  • Losses: {self.stats['losses']} ❌

{rr_emoji} <b>СУММА RR: {rr_sign}{total_rr:.2f}R</b>

━━━━━━━━━━━━━━━━━━

📍 <b>ОТКРЫТЫЕ</b> ({self.stats['open']})
{positions_text}

🎯 <b>ЧАСТИЧНО</b>: {self.stats.get('partial', 0)}
⏳ <b>ОЖИДАЮТ</b>: {self.stats['pending']}

━━━━━━━━━━━━━━━━━━
<i>Авто-обновление</i>"""

    def update_stats_from_tracker(self, tracker_stats: dict):
        """Синхронизирует статистику из position_tracker"""
        self.stats["pending"] = tracker_stats.get("pending", 0)
        self.stats["open"] = tracker_stats.get("open", 0)
        self.stats["partial"] = tracker_stats.get("partial", 0)
        self.stats["wins"] = tracker_stats.get("closed_tp", 0)
        self.stats["losses"] = tracker_stats.get("closed_sl", 0)
        self.stats["total_rr"] = tracker_stats.get("total_rr", 0.0)

    def add_active_position(self, symbol: str, direction: str, entry: float, partial: bool = False):
        """Добавляет позицию в список активных"""
        key = f"{symbol}_{direction}_{entry}"
        self.active_positions[key] = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "partial": partial
        }

    def remove_active_position(self, symbol: str, direction: str, entry: float):
        """Удаляет позицию из списка"""
        key = f"{symbol}_{direction}_{entry}"
        self.active_positions.pop(key, None)