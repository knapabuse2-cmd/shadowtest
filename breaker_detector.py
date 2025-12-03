from typing import List, Optional
from dataclasses import dataclass

from analysis_interfaces import Zone, DetectionResult
# Candle импортируем только типом - у тебя он уже есть в data_interfaces,
# но тут не обязателен. Главное, чтобы у свечи были .open/.high/.low/.close


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str  # "HIGH" или "LOW"


def _get_candle_timestamp(candle) -> Optional[int]:
    """Безопасно извлекает timestamp из свечи"""
    ts = getattr(candle, 'time', None)
    if ts is None:
        ts = getattr(candle, 'timestamp', None)

    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1000)

    if isinstance(ts, (int, float)):
        if ts < 1000000000000:
            return int(ts * 1000)
        return int(ts)

    import time
    return int(time.time() * 1000)

class BreakerDetector:
    """
    PRO-версия ICT Breaker Block Detector

    Логика (упрощённо, но по ICT):

    Bullish Breaker:
      1) Есть два свинга LOW: старый low1 и более свежий low2, причём low2 < low1 (sweep)
      2) После low2 формируется BOS UP (закрытие выше локальной структуры)
      3) Последняя медвежья свеча перед BOS UP = breaker candle
      4) Зона breaker = диапазон тела (можно учитывать и wick)

    Bearish Breaker:
      1) Два свинга HIGH: high1 и более свежий high2, причём high2 > high1 (sweep)
      2) После high2 BOS DOWN (закрытие ниже локальной структуры)
      3) Последняя бычья свеча перед BOS DOWN = breaker candle
      4) Зона breaker = диапазон тела
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        min_sweep_ratio: float = 0.0005,      # минимальный относительный "вынос"
        min_displacement_factor: float = 1.5  # фактор размера тела свечи для BOS
    ):
        self.swing_lookback = swing_lookback
        self.min_sweep_ratio = min_sweep_ratio
        self.min_displacement_factor = min_displacement_factor

    # --------------------------------------------------
    # PUBLIC API — будет вызываться Orchestrator'ом
    # detect(candles, tf) → DetectionResult
    # --------------------------------------------------

    def detect(self, candles: List, tf: str) -> DetectionResult:
        """ИСПРАВЛЕННЫЙ метод detect для BreakerDetector с timestamps"""
        zones: List[Zone] = []

        if not candles or len(candles) < self.swing_lookback * 4:
            return DetectionResult([], None)

        swings = self._find_swings(candles)

        bull_zone = self._detect_bullish_breaker(candles, swings)
        if bull_zone:
            bull_zone.tf = tf
            # ДОБАВЛЯЕМ timestamp и candle_index
            if len(candles) > 0:
                # Находим свечу breaker
                for i in range(len(candles) - 1, 0, -1):
                    c = candles[i]
                    if c.low >= bull_zone.low and c.high <= bull_zone.high:
                        bull_zone.timestamp = _get_candle_timestamp(c)
                        bull_zone.candle_index = i
                        break
                # Fallback на последнюю свечу
                if bull_zone.timestamp is None:
                    bull_zone.timestamp = _get_candle_timestamp(candles[-1])
                    bull_zone.candle_index = len(candles) - 1
            zones.append(bull_zone)

        bear_zone = self._detect_bearish_breaker(candles, swings)
        if bear_zone:
            bear_zone.tf = tf
            # ДОБАВЛЯЕМ timestamp и candle_index
            if len(candles) > 0:
                # Находим свечу breaker
                for i in range(len(candles) - 1, 0, -1):
                    c = candles[i]
                    if c.low >= bear_zone.low and c.high <= bear_zone.high:
                        bear_zone.timestamp = _get_candle_timestamp(c)
                        bear_zone.candle_index = i
                        break
                # Fallback на последнюю свечу
                if bear_zone.timestamp is None:
                    bear_zone.timestamp = _get_candle_timestamp(candles[-1])
                    bear_zone.candle_index = len(candles) - 1
            zones.append(bear_zone)

        return DetectionResult(zones, None)

    # --------------------------------------------------
    # SWINGS
    # --------------------------------------------------
    def _find_swings(self, candles: List) -> List[SwingPoint]:
        """
        Находим локальные HIGH/LOW по окну swing_lookback
        """
        swings: List[SwingPoint] = []
        n = len(candles)
        L = self.swing_lookback

        if n < 2 * L + 1:
            return swings

        for i in range(L, n - L):
            h = candles[i].high
            l = candles[i].low

            is_high = all(h >= candles[j].high for j in range(i - L, i + L + 1))
            is_low = all(l <= candles[j].low for j in range(i - L, i + L + 1))

            if is_high:
                swings.append(SwingPoint(i, h, "HIGH"))
            if is_low:
                swings.append(SwingPoint(i, l, "LOW"))

        return swings

    # --------------------------------------------------
    # BULLISH BREAKER
    # --------------------------------------------------
    def _detect_bullish_breaker(
        self,
        candles: List,
        swings: List[SwingPoint]
    ) -> Optional[Zone]:
        """
        Ищем pattern:
          LOW (low1) → LOW (swept low2 < low1) → BOS UP → последняя медвежья свеча перед BOS
        """
        # Отбираем только LOW swings
        lows = [s for s in swings if s.kind == "LOW"]
        if len(lows) < 2:
            return None

        # Берём последние несколько комбинаций
        for i in range(len(lows) - 1, 0, -1):
            low2 = lows[i]          # более свежий low
            low1 = lows[i - 1]      # предыдущий low

            if low2.index <= low1.index:
                continue

            # Проверяем sweep: low2 существенно ниже low1
            if not self._is_sweep(low1.price, low2.price, direction="DOWN"):
                continue

            # Ищем BOS UP после low2
            bos_index = self._find_bos_up(candles, low1.index, low2.index)
            if bos_index is None:
                continue

            # Находим breaker candle: последняя медвежья свеча перед BOS
            breaker_idx = self._last_bearish_before(candles, low2.index, bos_index)
            if breaker_idx is None:
                continue

            c = candles[breaker_idx]

            # Проверка на displacement (свеча BOS должна быть достаточно крупной)
            if not self._has_bullish_displacement(candles, bos_index):
                continue

            body_low = min(c.open, c.close)
            body_high = max(c.open, c.close)

            return Zone(
                tf="",
                low=body_low,
                high=body_high,
                type="BREAKER_BULL"
            )

        return None

    # --------------------------------------------------
    # BEARISH BREAKER
    # --------------------------------------------------
    def _detect_bearish_breaker(
        self,
        candles: List,
        swings: List[SwingPoint]
    ) -> Optional[Zone]:
        """
        Ищем pattern:
          HIGH (high1) → HIGH (swept high2 > high1) → BOS DOWN → последняя бычья свеча перед BOS
        """
        highs = [s for s in swings if s.kind == "HIGH"]
        if len(highs) < 2:
            return None

        for i in range(len(highs) - 1, 0, -1):
            high2 = highs[i]
            high1 = highs[i - 1]

            if high2.index <= high1.index:
                continue

            # Проверяем sweep: high2 значительно выше high1
            if not self._is_sweep(high1.price, high2.price, direction="UP"):
                continue

            bos_index = self._find_bos_down(candles, high1.index, high2.index)
            if bos_index is None:
                continue

            breaker_idx = self._last_bullish_before(candles, high2.index, bos_index)
            if breaker_idx is None:
                continue

            c = candles[breaker_idx]

            if not self._has_bearish_displacement(candles, bos_index):
                continue

            body_low = min(c.open, c.close)
            body_high = max(c.open, c.close)

            return Zone(
                tf="",
                low=body_low,
                high=body_high,
                type="BREAKER_BEAR"
            )

        return None

    # --------------------------------------------------
    # BOS / SWEEP / DISPLACEMENT HELPERS
    # --------------------------------------------------
    def _is_sweep(self, old_price: float, new_price: float, direction: str) -> bool:
        """
        Проверка, что есть sweep ликвидности:
        - DOWN: new_price < old_price * (1 - min_sweep_ratio)
        - UP:   new_price > old_price * (1 + min_sweep_ratio)
        """
        if direction == "DOWN":
            return new_price < old_price * (1 - self.min_sweep_ratio)
        else:
            return new_price > old_price * (1 + self.min_sweep_ratio)

    def _find_bos_up(self, candles: List, idx1: int, idx2: int) -> Optional[int]:
        """
        Ищем BOS UP после low2:
        close > max high в диапазоне [idx1, idx2] + небольшой запас
        """
        if idx2 >= len(candles) - 2:
            return None

        local_high = max(c.high for c in candles[idx1:idx2 + 1])
        threshold = local_high * 1.0001

        for i in range(idx2 + 1, len(candles)):
            if candles[i].close > threshold:
                return i
        return None

    def _find_bos_down(self, candles: List, idx1: int, idx2: int) -> Optional[int]:
        """
        BOS DOWN: close < min low в диапазоне [idx1, idx2]
        """
        if idx2 >= len(candles) - 2:
            return None

        local_low = min(c.low for c in candles[idx1:idx2 + 1])
        threshold = local_low * 0.9999

        for i in range(idx2 + 1, len(candles)):
            if candles[i].close < threshold:
                return i
        return None

    def _last_bearish_before(self, candles: List, start_idx: int, end_idx: int) -> Optional[int]:
        """
        Последняя медвежья свеча перед BOS UP
        """
        for i in range(end_idx - 1, start_idx - 1, -1):
            if candles[i].close < candles[i].open:
                return i
        return None

    def _last_bullish_before(self, candles: List, start_idx: int, end_idx: int) -> Optional[int]:
        """
        Последняя бычья свеча перед BOS DOWN
        """
        for i in range(end_idx - 1, start_idx - 1, -1):
            if candles[i].close > candles[i].open:
                return i
        return None

    def _avg_body(self, candles: List, lookback: int = 50) -> float:
        if not candles:
            return 0.0
        tail = candles[-lookback:]
        bodies = [abs(c.close - c.open) for c in tail]
        if not bodies:
            return 0.0
        return sum(bodies) / len(bodies)

    def _has_bullish_displacement(self, candles: List, idx: int) -> bool:
        """
        Проверка, что свеча BOS UP действительно displacement:
        - бычья
        - тело > avg_body * min_displacement_factor
        """
        c = candles[idx]
        if c.close <= c.open:
            return False
        avg_body = self._avg_body(candles)
        if avg_body <= 0:
            return False
        return (c.close - c.open) > avg_body * self.min_displacement_factor

    def _has_bearish_displacement(self, candles: List, idx: int) -> bool:
        """
        Аналогично для BOS DOWN
        """
        c = candles[idx]
        if c.close >= c.open:
            return False
        avg_body = self._avg_body(candles)
        if avg_body <= 0:
            return False
        return (c.open - c.close) > avg_body * self.min_displacement_factor
