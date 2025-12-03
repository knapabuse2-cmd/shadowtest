# config.py - Конфигурация из .env
import os
from dotenv import load_dotenv

# Загружаем .env файл
load_dotenv()

# --------------------------
#  TELEGRAM
# --------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
LOG_CHAT_ID = os.getenv("LOG_CHAT_ID")

# Проверка что токены загружены
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in .env file!")
if not CHAT_ID:
    raise ValueError("CHAT_ID not found in .env file!")

# --------------------------
#  TRADING SETTINGS
# --------------------------
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BCH/USDT", "BNB/USDT",
    "ADA/USDT", "ATOM/USDT", "APT/USDT", "ARB/USDT",
    "AVAX/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT",
    "NEAR/USDT", "SOL/USDT"
]

TIMEFRAMES = ["1d", "4h", "1h", "15m"]

# --------------------------
#  INTERVALS
# --------------------------
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", 60))
POSITION_CLEANUP_INTERVAL = int(os.getenv("POSITION_CLEANUP_INTERVAL", 100))
CORRELATION_UPDATE_INTERVAL = int(os.getenv("CORRELATION_UPDATE_INTERVAL", 20))

# --------------------------
#  FILTERS
# --------------------------
MIN_CONFLUENCE_SCORE = int(os.getenv("MIN_CONFLUENCE_SCORE", 30))

# --------------------------
#  FILES
# --------------------------
PERFORMANCE_DATA_FILE = os.getenv("PERFORMANCE_DATA_FILE", "performance_data.json")