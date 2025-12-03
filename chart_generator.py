# chart_generator_api.py
# =======================
# TradingView графики через chart-img.com API
# Просто HTTP запрос - никаких браузеров!

import aiohttp
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
import os

# API ключ (получить на chart-img.com)
CHART_IMG_API_KEY = os.getenv("CHART_IMG_API_KEY", "8qBHjCsGbNGg7uE46W19l1TxXVIrC834CZyNcf90")


@dataclass
class ZoneData:
    low: float
    high: float
    zone_type: str
    start_time: Optional[int] = None
    label: Optional[str] = None


@dataclass
class SignalData:
    symbol: str
    tf: str
    direction: str  # LONG / SHORT
    entry: float
    stop_loss: float
    take_profits: List[float]
    zones: List[ZoneData] = field(default_factory=list)
    chain_name: str = ""
    rr: float = 0.0


# Маппинг таймфреймов для chart-img.com
TF_MAP = {
    "1m": "1",
    "5m": "5", 
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

# Маппинг символов для TradingView
def get_tv_symbol(symbol: str) -> str:
    """Конвертирует символ в формат TradingView"""
    # BTC/USDT -> BINANCE:BTCUSDT
    clean = symbol.replace("/", "").replace("-", "")
    return f"BINANCE:{clean}"


class ChartGeneratorAPI:
    """
    Генератор графиков через chart-img.com API v2
    
    Плюсы:
    - Никаких зависимостей (только aiohttp)
    - Быстро (~1-2 сек)
    - Настоящий TradingView вид
    
    Минусы:
    - Нужен API ключ (бесплатный)
    - Лимит запросов (100/день бесплатно)
    """
    
    API_URL = "https://api.chart-img.com/v2/tradingview/advanced-chart"
    
    def __init__(self, api_key: str = None, width: int = 800, height: int = 600):
        self.api_key = api_key or CHART_IMG_API_KEY
        self.width = width
        self.height = height
        
        if not self.api_key:
            print("⚠️ CHART_IMG_API_KEY not set!")
            print("   Get free key at: https://chart-img.com")
    
    async def generate(self, signal: SignalData) -> Optional[bytes]:
        """
        Генерирует график через API.
        
        Возвращает PNG bytes или None при ошибке.
        """
        if not self.api_key:
            print("❌ No API key!")
            return None
        
        # Конвертируем символ и таймфрейм
        tv_symbol = get_tv_symbol(signal.symbol)
        tv_interval = TF_MAP.get(signal.tf.lower(), "60")
        
        # Определяем направление
        is_long = "LONG" in signal.direction.upper() or "BUY" in signal.direction.upper()
        
        # Базовый payload по документации chart-img.com v2
        payload = {
            "symbol": tv_symbol,
            "interval": tv_interval,
            "theme": "dark",
            "width": self.width,
            "height": self.height,
            "format": "png",
            # Индикаторы
            "studies": [
                {"name": "Volume"}
            ],
            # Рисунки (линии Entry/SL/TP)
            "drawings": self._build_drawings(signal, is_long),
        }
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        error = await resp.text()
                        print(f"❌ chart-img.com error {resp.status}: {error}")
                        return None
                        
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def _build_drawings(self, signal: SignalData, is_long: bool) -> list:
        """
        Строит массив drawings по формату chart-img.com v2
        
        Формат:
        {
            "name": "horizontal_line",
            "input": {
                "price": 123.45
            },
            "options": { ... }
        }
        """
        drawings = []
        
        # Entry линия (белая)
        drawings.append({
            "name": "horizontal_line",
            "input": {
                "price": signal.entry
            },
            "options": {
                "lineColor": "#FFFFFF",
                "lineWidth": 2,
                "lineStyle": 2,
                "showLabel": True,
                "text": f"ENTRY {signal.entry:.2f}",
            }
        })
        
        # Stop Loss (красная)
        drawings.append({
            "name": "horizontal_line",
            "input": {
                "price": signal.stop_loss
            },
            "options": {
                "lineColor": "#EF5350",
                "lineWidth": 2,
                "lineStyle": 2,
                "showLabel": True,
                "text": f"SL {signal.stop_loss:.2f}",
            }
        })
        
        # Take Profits (зелёные)
        for i, tp in enumerate(signal.take_profits[:2]):
            drawings.append({
                "name": "horizontal_line",
                "input": {
                    "price": tp
                },
                "options": {
                    "lineColor": "#26A69A",
                    "lineWidth": 2,
                    "lineStyle": 2,
                    "showLabel": True,
                    "text": f"TP{i+1} {tp:.2f}",
                }
            })
        
        return drawings


class ChartGenerator:
    """
    Обёртка для совместимости с существующим кодом.
    """
    
    def __init__(self, api_key: str = None, width: int = 800, height: int = 600, **kwargs):
        self.api_generator = ChartGeneratorAPI(api_key, width, height)
    
    def generate(self, candles, signal: SignalData, **kwargs) -> Optional[bytes]:
        """
        Синхронная обёртка.
        candles игнорируются - API сам берёт данные с TradingView.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.api_generator.generate(signal))
        finally:
            loop.close()
    
    async def generate_async(self, signal: SignalData) -> Optional[bytes]:
        """Асинхронная генерация"""
        return await self.api_generator.generate(signal)


# Проверка доступности
CHARTS_AVAILABLE = True


# Тест
if __name__ == "__main__":
    # Тестовый сигнал
    signal = SignalData(
        symbol="BTC/USDT",
        tf="1h",
        direction="LONG",
        entry=96500,
        stop_loss=95800,
        take_profits=[97500, 98500],
        zones=[],
        chain_name="Liquidity Sweep",
        rr=2.5,
    )
    
    print(f"🔍 Generating chart for {signal.symbol} {signal.tf}...")
    print(f"   Entry: {signal.entry}, SL: {signal.stop_loss}, TP: {signal.take_profits}")
    
    async def test():
        gen = ChartGeneratorAPI(width=800, height=600)
        result = await gen.generate(signal)
        
        if result:
            out_path = "test_chart_api.png"
            with open(out_path, "wb") as f:
                f.write(result)
            print(f"✅ Chart saved to {out_path} ({len(result)} bytes)")
        else:
            print("❌ Failed to generate chart")
    
    asyncio.run(test())