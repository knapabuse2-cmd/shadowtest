# utils.py
# ========================================
# Общие утилиты для всего проекта
# Убирает дублирование кода в детекторах
# ========================================

from typing import Optional, List, Dict, Any
from datetime import datetime
import time


def get_candle_timestamp(candle) -> Optional[int]:
    """
    Безопасно извлекает timestamp из свечи.
    Поддерживает разные форматы: time, timestamp, datetime объекты.
    
    Returns:
        Timestamp в миллисекундах или None
    """
    # Пробуем разные варианты получения времени
    ts = getattr(candle, 'time', None)
    if ts is None:
        ts = getattr(candle, 'timestamp', None)

    # Если это datetime объект
    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1000)

    # Если это уже число
    if isinstance(ts, (int, float)):
        # Проверяем, не в секундах ли (слишком маленькое число)
        if ts < 1000000000000:  # Меньше чем timestamp в миллисекундах
            return int(ts * 1000)
        return int(ts)

    # Fallback на текущее время
    return int(time.time() * 1000)


def avg_body(candles: List, lookback: int = 20) -> float:
    """
    Средний размер тела свечи за lookback.
    """
    if not candles:
        return 0.0

    sub = candles[-lookback:] if len(candles) > lookback else candles
    bodies = [abs(c.close - c.open) for c in sub]
    if not bodies:
        return 0.0

    return sum(bodies) / len(bodies)


def calculate_atr(candles: List, period: int = 14) -> float:
    """
    Рассчитывает ATR (Average True Range).
    """
    if len(candles) < 2:
        return 0.0

    trs = []
    for i in range(1, min(period + 1, len(candles))):
        h = candles[-i].high
        l = candles[-i].low
        prev_close = candles[-i - 1].close
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)

    return sum(trs) / len(trs) if trs else 0.0


def format_price(price: float, symbol: str = "") -> str:
    """
    Форматирует цену с правильным количеством знаков.
    """
    if price is None:
        return "N/A"

    if symbol and ("BTC" in symbol.upper() or "ETH" in symbol.upper()):
        if price > 100:
            return f"{price:,.2f}"
        return f"{price:.2f}"
    elif price > 1:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def normalize_direction(direction) -> str:
    """
    Нормализует направление к LONG/SHORT.
    """
    dir_str = str(direction).upper()
    dir_str = dir_str.replace("DIRECTION.", "").replace("MARKETBIAS.", "")
    
    if dir_str in ["LONG", "BUY"]:
        return "LONG"
    elif dir_str in ["SHORT", "SELL"]:
        return "SHORT"
    return dir_str


def zones_overlap(z1_low: float, z1_high: float, z2_low: float, z2_high: float, tolerance: float = 0.001) -> bool:
    """
    Проверка пересечения двух зон с допуском.
    """
    z1_expanded_high = z1_high * (1 + tolerance)
    z1_expanded_low = z1_low * (1 - tolerance)
    
    return not (z1_expanded_high < z2_low or z1_expanded_low > z2_high)


def find_swings(candles: List, left: int = 3, right: int = 2) -> List[Dict]:
    """
    Находит swing highs и swing lows.
    
    Returns:
        List[{"kind": "HIGH"/"LOW", "index": int, "price": float}]
    """
    swings = []
    n = len(candles)
    
    if n < left + right + 1:
        return swings
    
    for i in range(left, n - right):
        c = candles[i]
        
        # Swing High
        is_high = all(c.high >= candles[j].high for j in range(i - left, i + right + 1) if j != i)
        
        # Swing Low  
        is_low = all(c.low <= candles[j].low for j in range(i - left, i + right + 1) if j != i)
        
        if is_high:
            swings.append({
                "kind": "HIGH",
                "index": i,
                "price": c.high,
                "time": get_candle_timestamp(c)
            })
        if is_low:
            swings.append({
                "kind": "LOW", 
                "index": i,
                "price": c.low,
                "time": get_candle_timestamp(c)
            })
    
    return swings
