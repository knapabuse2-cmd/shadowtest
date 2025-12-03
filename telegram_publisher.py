# telegram_publisher.py
# =====================
# Telegram публикатор с РУССКИМИ объяснениями для СНГ аудитории
# Кодировка: UTF-8, но без проблемных символов

import aiohttp
from typing import List, Optional
from analysis_interfaces import ChainSignal

# ---------------------------------------------------------------
#    СЛОВАРИ ПЕРЕВОДОВ И ОБЪЯСНЕНИЙ
# ---------------------------------------------------------------

CHAIN_DESCRIPTIONS_RU = {
    "1.1": {
        "name": "Multi-TF Confluence",
        "desc": "Сетап с подтверждением на 4 таймфреймах: Daily + 4H + 1H + 15m",
        "logic": "Daily OB/FVG задает направление, 4H зона внутри Daily, вход на 15m FVG внутри 1H зоны",
        "strength": "Очень сильный",
    },
    "1.2": {
        "name": "IDM Cascade",
        "desc": "Каскад индуцементов (ловушек) на Daily и 4H",
        "logic": "Smart Money собирают ликвидность на D и 4H (IDM), затем разворот. Вход на 15m",
        "strength": "Сильный",
    },
    "1.3": {
        "name": "Daily POI + 4H IDM",
        "desc": "Daily зона + ловушка на 4H перед продолжением",
        "logic": "Цена в Daily OB/FVG, на 4H был IDM (сбор стопов), вход на 15m",
        "strength": "Сильный",
    },
    "1.4": {
        "name": "Daily FVG Fill",
        "desc": "Заполнение Daily имбаланса с подтверждением на младших TF",
        "logic": "Daily FVG как магнит, 4H зона внутри, вход на 15m после теста 1H",
        "strength": "Средний+",
    },
    "1.5": {
        "name": "FVG Reaction",
        "desc": "Реакция от Daily FVG с подтверждением на 4H",
        "logic": "Цена вернулась в Daily FVG (первый тест), ищем 4H FVG для входа",
        "strength": "Средний",
    },
    "2.6": {
        "name": "Liquidity Sweep",
        "desc": "Снятие ликвидности на 4H с входом на младшем TF",
        "logic": "Фрактал 4H пробит (sweep), цена вернулась в POI, вход на 15m FVG",
        "strength": "Очень сильный",
    },
    "3.2": {
        "name": "First Touch FVG",
        "desc": "Первый заход в 4H FVG (свежая зона)",
        "logic": "4H FVG еще не тестировался - первое касание самое сильное. Вход на 1H FVG",
        "strength": "Сильный",
    },
    "Signal_1": {
        "name": "FH + OB Combo",
        "desc": "Фрактальный хай/лоу на Daily + Order Block на 4H",
        "logic": "Daily показал разворотную структуру (FH), на 4H сформировался OB для входа",
        "strength": "Средний",
    },
}

DIRECTION_RU = {
    "LONG": "ЛОНГ (покупка)",
    "SHORT": "ШОРТ (продажа)",
    "BUY": "ЛОНГ (покупка)",
    "SELL": "ШОРТ (продажа)",
}

BIAS_RU = {
    "STRONG_BULLISH": "Сильный бычий тренд",
    "BULLISH": "Бычий тренд",
    "RANGE": "Боковик/консолидация",
    "BEARISH": "Медвежий тренд",
    "STRONG_BEARISH": "Сильный медвежий тренд",
}

TERMS_RU = {
    "OB": "Order Block (блок ордеров)",
    "FVG": "Fair Value Gap (имбаланс)",
    "IDM": "Inducement (ловушка/сбор стопов)",
    "FH": "Fractal High/Low (фрактал)",
    "POI": "Point of Interest (зона интереса)",
    "BSL": "Buy Side Liquidity (ликвидность покупателей)",
    "SSL": "Sell Side Liquidity (ликвидность продавцов)",
    "EQH": "Equal Highs (равные хаи)",
    "EQL": "Equal Lows (равные лои)",
    "BOS": "Break of Structure (слом структуры)",
    "CHoCH": "Change of Character (смена характера)",
    "MSS": "Market Structure Shift (сдвиг структуры)",
}


# ---------------------------------------------------------------
#    HELPER FUNCTIONS
# ---------------------------------------------------------------

def _clean_direction(direction) -> str:
    """Очищает направление от enum wrapper"""
    d = str(direction).upper()
    d = d.replace("DIRECTION.", "").replace("MARKETBIAS.", "")
    return d


def _clean_bias(bias) -> str:
    """Очищает bias от enum wrapper"""
    if bias is None:
        return ""
    b = str(bias).upper()
    b = b.replace("MARKETBIAS.", "").replace("BIAS.", "")
    # Убираем числа в скобках типа "(40)"
    if "(" in b:
        b = b.split("(")[0].strip()
    return b


def _format_price(price: float, symbol: str = "") -> str:
    """Форматирует цену с правильным количеством знаков"""
    if price is None:
        return "N/A"

    # BTC/ETH - 2 знака, альты - больше
    if symbol and ("BTC" in symbol.upper() or "ETH" in symbol.upper()):
        if price > 100:
            return f"{price:,.2f}"
        return f"{price:.2f}"
    elif price > 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def _get_chain_info(chain_id: str) -> dict:
    """Получает информацию о цепочке"""
    # Нормализуем ID
    normalized = chain_id.replace("Chain_", "").replace("chain_", "")

    if normalized in CHAIN_DESCRIPTIONS_RU:
        return CHAIN_DESCRIPTIONS_RU[normalized]

    # Fallback
    return {
        "name": chain_id,
        "desc": "ICT/SMC сетап",
        "logic": "Multi-timeframe analysis",
        "strength": "Средний",
    }


def _calculate_rr_info(entry: float, sl: float, tps: List[float]) -> str:
    """Рассчитывает и объясняет RR"""
    if not entry or not sl or not tps:
        return ""

    risk = abs(entry - sl)
    if risk == 0:
        return ""

    lines = []
    for i, tp in enumerate(tps[:3], 1):
        reward = abs(tp - entry)
        rr = reward / risk
        pnl_percent = (reward / entry) * 100
        lines.append(f"  TP{i}: RR {rr:.1f}x (+{pnl_percent:.2f}%)")

    return "\n".join(lines)


def _get_sl_explanation(signal: ChainSignal) -> str:
    """Объясняет почему стоп там где он есть"""
    desc = signal.description.lower() if signal.description else ""

    if "structural" in desc or "swing" in desc:
        return "За структурным свингом (защита от манипуляций)"
    elif "poi" in desc:
        return "За POI зоной старшего TF"
    else:
        return "За ключевым уровнем"


def _get_tp_explanation(signal: ChainSignal) -> str:
    """Объясняет цели"""
    desc = signal.description.lower() if signal.description else ""

    if "liquidity" in desc:
        return "На уровнях ликвидности (EQH/EQL)"
    elif "imbalance" in desc:
        return "На незакрытых имбалансах"
    else:
        return "На ключевых уровнях сопротивления/поддержки"


# ---------------------------------------------------------------
#    TELEGRAM PUBLISHER
# ---------------------------------------------------------------

class TelegramPublisher:
    """
    Telegram публикатор с подробными русскими объяснениями.
    Поддерживает отправку графиков.
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.session: aiohttp.ClientSession | None = None
        self.url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        # Chart generator (ленивая инициализация)
        self._chart_generator = None

    @property
    def chart_generator(self):
        """Ленивая инициализация генератора графиков"""
        if self._chart_generator is None:
            try:
                from chart_generator import ChartGenerator
                self._chart_generator = ChartGenerator()
            except ImportError:
                self._chart_generator = None
        return self._chart_generator

    async def connect(self):
        self.session = aiohttp.ClientSession()

    async def disconnect(self):
        if self.session:
            await self.session.close()
            self.session = None

    def _format_signal(self, s: ChainSignal) -> str:
        """
        Форматирует сигнал с подробными объяснениями на русском.
        """
        # Получаем информацию о цепочке
        chain_info = _get_chain_info(s.chain_id)

        # Направление
        direction_clean = _clean_direction(s.direction)
        direction_ru = DIRECTION_RU.get(direction_clean, direction_clean)

        # Эмодзи направления
        if "LONG" in direction_clean or "BUY" in direction_clean:
            dir_emoji = "🟢"
            dir_arrow = "^"
        else:
            dir_emoji = "🔴"
            dir_arrow = "v"

        # Сила сетапа
        strength = chain_info.get("strength", "Средний")
        if "Очень" in strength:
            strength_emoji = "💎💎💎"
        elif "Сильный" in strength:
            strength_emoji = "💎💎"
        else:
            strength_emoji = "💎"

        # Форматируем цены
        entry_str = _format_price(s.entry, s.symbol)
        sl_str = _format_price(s.stop_loss, s.symbol)

        # TP строки
        tp_lines = []
        if s.take_profits:
            for i, tp in enumerate(s.take_profits[:3], 1):
                tp_lines.append(f"  TP{i}: {_format_price(tp, s.symbol)}")

        # RR расчет
        risk_pct = ""
        if s.entry and s.stop_loss:
            risk = abs(s.entry - s.stop_loss) / s.entry * 100
            risk_pct = f" ({risk:.2f}%)"

        # Собираем сообщение
        msg = f"""
{dir_emoji} <b>ICT/SMC СИГНАЛ</b> {dir_emoji}

<b>{s.symbol}</b> | {s.tf.upper()} | {direction_ru}

{'=' * 30}
<b>СЕТАП:</b> {chain_info['name']}
{strength_emoji} Сила: {strength}

<b>Логика:</b>
{chain_info['logic']}

{'=' * 30}
<b>ТОЧКИ ВХОДА:</b>

{dir_arrow} Entry: <code>{entry_str}</code>
{chr(10).join(tp_lines)}

<b>СТОП:</b> <code>{sl_str}</code>{risk_pct}
{_get_sl_explanation(s)}

<b>RR:</b> {s.rr:.1f}x
"""

        # Добавляем расчет RR для каждого TP
        if s.take_profits and s.entry and s.stop_loss:
            rr_info = _calculate_rr_info(s.entry, s.stop_loss, s.take_profits)
            if rr_info:
                msg += f"\n{rr_info}\n"

        # Объяснение целей
        msg += f"""
<b>Цели:</b> {_get_tp_explanation(s)}

{'=' * 30}
<b>ОПИСАНИЕ СЕТАПА:</b>
{chain_info['desc']}
"""

        # Bias если есть
        if hasattr(s, 'bias') and s.bias:
            bias_clean = _clean_bias(s.bias)
            bias_ru = BIAS_RU.get(bias_clean, bias_clean)
            msg += f"\n<b>Рыночный контекст:</b> {bias_ru}"

        # Риск-менеджмент совет
        msg += f"""

{'=' * 30}
<b>РИСК-МЕНЕДЖМЕНТ:</b>
- Риск на сделку: 1-2% депозита
- Частичная фиксация на TP1 (50%)
- Стоп в безубыток после TP1
- Трейлинг остатка к TP2/TP3

<i>NFA. Всегда делайте собственный анализ.</i>
"""

        return msg.strip()

    def _format_signal_compact(self, s: ChainSignal) -> str:
        """
        Компактный формат для быстрого чтения.
        """
        chain_info = _get_chain_info(s.chain_id)
        direction_clean = _clean_direction(s.direction)

        if "LONG" in direction_clean or "BUY" in direction_clean:
            dir_emoji = "🟢"
        else:
            dir_emoji = "🔴"

        entry_str = _format_price(s.entry, s.symbol)
        sl_str = _format_price(s.stop_loss, s.symbol)

        tp_str = ""
        if s.take_profits:
            tp_str = " | ".join(_format_price(tp, s.symbol) for tp in s.take_profits[:2])

        msg = f"""{dir_emoji} <b>{s.symbol}</b> {direction_clean}

<b>Сетап:</b> {chain_info['name']} ({s.tf})
<b>Вход:</b> <code>{entry_str}</code>
<b>Стоп:</b> <code>{sl_str}</code>
<b>Цели:</b> {tp_str}
<b>RR:</b> {s.rr:.1f}x

<i>{chain_info['desc']}</i>
"""
        return msg.strip()

    async def publish(self, signal: ChainSignal, compact: bool = False):
        """Публикует сигнал в Telegram"""
        if not self.session:
            raise RuntimeError("TelegramPublisher not connected")

        if compact:
            text = self._format_signal_compact(signal)
        else:
            text = self._format_signal(signal)

        try:
            async with self.session.post(
                    self.url,
                    data={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    },
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"Telegram error: {resp.status} - {error_text}")
                    return False
                return True
        except Exception as e:
            print(f"Telegram publish error: {e}")
            return False

    async def publish_batch(self, signals: List[ChainSignal], compact: bool = False):
        """Публикует список сигналов"""
        results = []
        for s in signals:
            result = await self.publish(s, compact=compact)
            results.append(result)
        return results

    async def publish_with_chart(
            self,
            signal: ChainSignal,
            candles: List,
            zones: List = None,
            compact_text: bool = True
    ):
        """
        Публикует сигнал с графиком.

        Args:
            signal: Сигнал
            candles: Список свечей для графика
            zones: Список зон для отрисовки
            compact_text: Использовать компактный текст
        """
        if not self.session:
            raise RuntimeError("TelegramPublisher not connected")

        # Генерируем график
        chart_bytes = None
        if self.chart_generator and candles:
            try:
                from chart_generator import SignalData, ZoneData
                import pandas as pd
                from datetime import datetime

                # Конвертируем свечи в DataFrame
                data = []
                for c in candles[-100:]:  # Последние 100 свечей
                    data.append({
                        'Date': getattr(c, 'time', datetime.now()),
                        'Open': float(c.open),
                        'High': float(c.high),
                        'Low': float(c.low),
                        'Close': float(c.close),
                        'Volume': float(getattr(c, 'volume', 0)),
                    })

                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Конвертируем зоны
                zone_data = []
                if zones:
                    for z in zones[-5:]:  # Последние 5 зон
                        zone_data.append(ZoneData(
                            low=z.low,
                            high=z.high,
                            start_idx=max(0, len(df) - 30),
                            end_idx=None,
                            zone_type=z.type,
                            label=z.type.split('_')[0] if '_' in z.type else z.type
                        ))

                # Получаем имя цепочки
                chain_info = _get_chain_info(signal.chain_id)

                # Создаем SignalData
                signal_data = SignalData(
                    symbol=signal.symbol,
                    tf=signal.tf,
                    direction=str(signal.direction).replace("Direction.", "").replace("DIRECTION.", ""),
                    entry=float(signal.entry),
                    stop_loss=float(signal.stop_loss),
                    take_profits=[float(tp) for tp in (signal.take_profits or [])],
                    zones=zone_data,
                    chain_name=chain_info.get('name', signal.chain_id),
                    rr=float(signal.rr or 0.0)
                )

                # Генерируем график
                chart_bytes = self.chart_generator.generate(df, signal_data)

            except Exception as e:
                print(f"Chart generation error: {e}")
                chart_bytes = None

        # Отправляем фото с подписью
        if chart_bytes:
            caption = self._format_signal_compact(signal) if compact_text else self._format_signal(signal)

            # Обрезаем подпись если слишком длинная (Telegram limit: 1024)
            if len(caption) > 1024:
                caption = caption[:1020] + "..."

            try:
                import aiohttp
                form_data = aiohttp.FormData()
                form_data.add_field('chat_id', self.chat_id)
                form_data.add_field('photo', chart_bytes, filename='chart.png', content_type='image/png')
                form_data.add_field('caption', caption)
                form_data.add_field('parse_mode', 'HTML')

                async with self.session.post(self.photo_url, data=form_data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"Telegram photo error: {resp.status} - {error_text}")
                        # Fallback: отправляем только текст
                        return await self.publish(signal, compact=compact_text)
                    return True
            except Exception as e:
                print(f"Telegram photo send error: {e}")
                return await self.publish(signal, compact=compact_text)
        else:
            # Нет графика - отправляем только текст
            return await self.publish(signal, compact=compact_text)

    async def publish_batch_with_charts(
            self,
            signals: List[ChainSignal],
            candles_dict: dict,  # symbol -> candles
            zones_dict: dict = None,  # symbol -> zones
            compact_text: bool = True
    ):
        """Публикует список сигналов с графиками"""
        results = []
        for s in signals:
            candles = candles_dict.get(s.symbol, [])
            zones = zones_dict.get(s.symbol, []) if zones_dict else []
            result = await self.publish_with_chart(s, candles, zones, compact_text)
            results.append(result)
        return results

    async def publish_summary(self, signals: List[ChainSignal]):
        """Публикует сводку по нескольким сигналам"""
        if not signals:
            return

        if not self.session:
            raise RuntimeError("TelegramPublisher not connected")

        # Группируем по направлению
        longs = [s for s in signals if
                 "LONG" in _clean_direction(s.direction) or "BUY" in _clean_direction(s.direction)]
        shorts = [s for s in signals if
                  "SHORT" in _clean_direction(s.direction) or "SELL" in _clean_direction(s.direction)]

        msg = f"""
📊 <b>СВОДКА СИГНАЛОВ</b>

Всего: {len(signals)} сигналов
🟢 Лонги: {len(longs)}
🔴 Шорты: {len(shorts)}

<b>Лучшие по RR:</b>
"""
        # Топ 5 по RR
        sorted_signals = sorted(signals, key=lambda x: x.rr or 0, reverse=True)[:5]
        for i, s in enumerate(sorted_signals, 1):
            dir_emoji = "🟢" if "LONG" in _clean_direction(s.direction) else "🔴"
            msg += f"{i}. {dir_emoji} {s.symbol} ({s.tf}) - RR {s.rr:.1f}x\n"

        msg += "\n<i>Подробности по каждому сигналу выше.</i>"

        await self.session.post(
            self.url,
            data={
                "chat_id": self.chat_id,
                "text": msg.strip(),
                "parse_mode": "HTML",
            },
        )


# ---------------------------------------------------------------
#    QUICK TEST
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Тест форматирования
    from dataclasses import dataclass


    @dataclass
    class TestSignal:
        symbol: str = "BTCUSDT"
        chain_id: str = "2.6"
        tf: str = "15m"
        direction: str = "LONG"
        entry: float = 95000.0
        stop_loss: float = 94200.0
        take_profits: list = None
        rr: float = 2.5
        description: str = "Chain 2.6: Liquidity sweep->POI->15m FVG | SL: Structural | TP: Liquidity"
        bias: str = "STRONG_BULLISH"

        def __post_init__(self):
            if self.take_profits is None:
                self.take_profits = [96000.0, 97500.0]


    pub = TelegramPublisher("test", "test")
    test_signal = TestSignal()

    print("=" * 50)
    print("ПОЛНЫЙ ФОРМАТ:")
    print("=" * 50)
    print(pub._format_signal(test_signal))

    print("\n" + "=" * 50)
    print("КОМПАКТНЫЙ ФОРМАТ:")
    print("=" * 50)
    print(pub._format_signal_compact(test_signal))