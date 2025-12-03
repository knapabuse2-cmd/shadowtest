import aiohttp

class TelegramLogPublisher:
    """
    Паблишер для логов (длинных сообщений).
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.token = bot_token
        self.chat_id = chat_id
        self.api = f"https://api.telegram.org/bot{self.token}/sendMessage"

    async def send_log(self, text: str):
        """
        Отправляет лог в отдельный телеграм канал.
        Сообщение автоматически разбивается, если слишком длинное.
        """

        # Telegram ограничение — 4096 символов
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]

        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                await session.post(
                    self.api,
                    data={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "parse_mode": "HTML"
                    }
                )
