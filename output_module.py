import aiohttp
from analysis_interfaces import ChainSignal
from typing import Optional
from position_tracker import TrackedPosition


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

        # Для закреплённого сообщения
        self.pinned_message_id: Optional[int] = None
        self.active_positions: dict = {}
        self.stats = {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "open": 0,
        }

    async def _send(self, text: str):
        """
        Внутренний метод отправки сообщения в Telegram
        """
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"❌ Failed to send to Telegram: HTTP {resp.status} → {body}")
        except Exception as e:
            print(f"❌ Error sending to Telegram: {e}")

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
        await self._send(text)

    async def publish_position_closed(self, pos: TrackedPosition, hit_tp_index: Optional[int]):
        """
        Сообщение о TP/SL.
        """
        direction_str = str(pos.direction).replace("Direction.", "").replace("DIRECTION.", "")

        if pos.outcome == "SL":
            icon = "🔴"
            outcome_text = "<b>Стоп-лосс сработал</b>"
        else:
            icon = "🟢"
            if hit_tp_index is not None:
                outcome_text = f"<b>Take Profit {hit_tp_index + 1} достигнут!</b>"
            else:
                outcome_text = "<b>Позиция закрыта</b>"

        text = (
            f"{icon} <b>ПОЗИЦИЯ ЗАКРЫТА</b>\n\n"
            f"<b>{pos.symbol}</b> | Chain <b>{pos.chain_id}</b>\n"
            f"Результат: {outcome_text}\n"
            f"Направление: <b>{direction_str}</b>\n"
            f"Entry: <b>{pos.entry:.5f}</b>\n"
            f"SL: <b>{pos.stop_loss:.5f}</b>\n"
            f"TPs: <b>{', '.join(f'{x:.5f}' for x in pos.take_profits)}</b>"
        )
        await self._send(text)

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
        await self._send(text)

    def _fmt_price(self, p: float) -> str:
        """Форматирует цену в зависимости от величины"""
        if p >= 1000:
            return f"{p:,.2f}"
        elif p >= 1:
            return f"{p:.4f}"
        else:
            return f"{p:.6f}"

    async def publish(self, signal: ChainSignal):
        """
        Отправка нового сигнала в Telegram с русскими объяснениями.
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
            final_rr = tp_rr  # Последний TP = финальный RR

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

        await self._send(text)

        # Увеличиваем счётчик сигналов
        self.stats["total_signals"] += 1

        # Обновляем закреп после нового сигнала
        if self.pinned_message_id:
            await self.update_pinned_stats()

    # ==========================================
    #  ЗАКРЕПЛЁННОЕ СООБЩЕНИЕ СО СТАТИСТИКОЙ
    # ==========================================

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
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", {}).get("message_id")
        except:
            pass
        return None

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
        """Формирует текст статистики"""
        from datetime import datetime
        now = datetime.now()

        total_closed = self.stats["wins"] + self.stats["losses"]
        win_rate = f"{(self.stats['wins'] / total_closed * 100):.1f}%" if total_closed > 0 else "—"

        # Список активных позиций
        pos_lines = []
        for key, pos in list(self.active_positions.items())[:10]:
            emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
            pos_lines.append(f"  {emoji} <b>{pos['symbol']}</b> @ {pos['entry']:.2f}")

        positions_text = "\n".join(pos_lines) if pos_lines else "  <i>Нет активных</i>"

        return f"""📊 <b>ICT/SMC BOT STATUS</b>
━━━━━━━━━━━━━━━━━━

🕐 Обновлено: {now.strftime("%H:%M:%S")}

📈 <b>СТАТИСТИКА</b>
  • Всего сигналов: {self.stats['total_signals']}
  • Win Rate: <b>{win_rate}</b>
  • Wins: {self.stats['wins']} ✅
  • Losses: {self.stats['losses']} ❌

━━━━━━━━━━━━━━━━━━

📍 <b>ОТКРЫТЫЕ</b> ({self.stats['open']})
{positions_text}

⏳ <b>ОЖИДАЮТ</b>: {self.stats['pending']}

━━━━━━━━━━━━━━━━━━
<i>Авто-обновление</i>"""

    def update_stats_from_tracker(self, tracker_stats: dict):
        """Синхронизирует статистику из position_tracker"""
        self.stats["pending"] = tracker_stats.get("pending", 0)
        self.stats["open"] = tracker_stats.get("open", 0)
        self.stats["wins"] = tracker_stats.get("closed_tp", 0)
        self.stats["losses"] = tracker_stats.get("closed_sl", 0)

    def add_active_position(self, symbol: str, direction: str, entry: float):
        """Добавляет позицию в список активных"""
        key = f"{symbol}_{direction}_{entry}"
        self.active_positions[key] = {
            "symbol": symbol,
            "direction": direction,
            "entry": entry
        }

    def remove_active_position(self, symbol: str, direction: str, entry: float):
        """Удаляет позицию из списка"""
        key = f"{symbol}_{direction}_{entry}"
        self.active_positions.pop(key, None)