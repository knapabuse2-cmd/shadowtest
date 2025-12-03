from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

    @property
    def time(self):
        """
        Конвертация времени в datetime (при необходимости).
        """
        import datetime
        return datetime.datetime.utcfromtimestamp(self.timestamp / 1000)


class IDataSource(Protocol):
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 300) -> List[Candle]:
        ...

    async def close(self):
        ...
