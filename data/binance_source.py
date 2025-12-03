import ccxt.async_support as ccxt
from dataclasses import dataclass


@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceDataSource:
    """
    Асинхронный источник данных через CCXT.
    """

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
        })

        # Приведение TF проекта → TF Binance
        self.tf_map = {
            "1d": "1d",
            "4h": "4h",
            "1h": "1h",
            "15m": "15m",
        }

    async def get_ohlcv(self, symbol: str, tf: str, limit: int = 300):
        """
        Основной метод → вызывается в orchestrator.
        Возвращает список Candle.
        """

        if tf not in self.tf_map:
            raise ValueError(f"Unsupported TF: {tf}")

        binance_tf = self.tf_map[tf]

        try:
            raw = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=binance_tf,
                limit=limit
            )
        except Exception as e:
            raise RuntimeError(f"Binance fetch error: {e}")

        candles = []
        for t, o, h, l, c, v in raw:
            candles.append(
                Candle(
                    time=t,
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v),
                )
            )

        return candles

    async def close(self):
        """Корректное закрытие CCXT."""
        try:
            await self.exchange.close()
        except Exception:
            pass
