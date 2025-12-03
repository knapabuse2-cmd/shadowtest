# news_filter.py
# ИСПРАВЛЕНО: Добавлены все необходимые импорты

from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import aiohttp

from analysis_interfaces import ChainSignal


class NewsEventFilter:
    """
    Блокирует сигналы во время важных новостей
    """

    def __init__(self):
        self.high_impact_events: List[Dict] = []
        self.last_update: Optional[datetime] = None
        self.block_before_minutes = 30  # Блокируем за 30 мин до новости
        self.block_after_minutes = 30   # Блокируем 30 мин после

    async def update_calendar(self):
        """
        Загружает календарь новостей (можно использовать API forexfactory)
        """
        # Это пример, нужен реальный API
        pass

    def should_block_signal(
            self,
            signal: ChainSignal,
            current_time: datetime
    ) -> Tuple[bool, str]:
        """
        Проверяет, не попадаем ли на важные новости
        """
        for event in self.high_impact_events:
            event_time = event['time']

            # Временное окно блокировки
            block_start = event_time - timedelta(minutes=self.block_before_minutes)
            block_end = event_time + timedelta(minutes=self.block_after_minutes)

            if block_start <= current_time <= block_end:
                return True, f"High impact news: {event['title']}"

        return False, "OK"
