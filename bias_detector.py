# bias_detector.py

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from analysis_interfaces import Zone, DetectionResult, VolumeContext
from data.data_interfaces import Candle


class MarketBias(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class BiasContext:
    bias: MarketBias
    htf_bias: Optional[MarketBias]  # Bias старшего TF
    structure_break: Optional[str]  # "BOS_UP", "BOS_DOWN", "CHoCH_UP", "CHoCH_DOWN"
    premium_discount: Optional[str]  # "PREMIUM", "EQUILIBRIUM", "DISCOUNT"
    order_flow: Optional[str]  # "BULLISH_OF", "BEARISH_OF", "NEUTRAL_OF"
    strength: float  # 0-100
    key_levels: Dict[str, float]  # "resistance", "support", "equilibrium"
    notes: List[str]


class ICTBiasDetector:
    """
    Продвинутое определение BIAS по методологии ICT
    """

    def __init__(self):
        self.swing_lookback = 10
        self.structure_lookback = 50
        self.htf_weight = 0.4  # Вес старшего TF в общем bias
        self.ltf_weight = 0.3  # Вес младшего TF
        self.of_weight = 0.3  # Вес order flow

    def detect_comprehensive_bias(
            self,
            candles_current: List[Candle],
            candles_htf: Optional[List[Candle]] = None,
            zones_current: Optional[List[Zone]] = None,
            tf_current: str = "1h",
            tf_htf: str = "4h"
    ) -> BiasContext:
        """
        Комплексное определение BIAS с учетом:
        1. Market Structure (BOS/CHoCH)
        2. Premium/Discount анализ
        3. Order Flow через зоны
        4. HTF подтверждение
        """

        notes = []

        # 1. MARKET STRUCTURE ANALYSIS
        structure_bias, structure_break = self._analyze_market_structure(candles_current)
        if structure_break:
            notes.append(f"Structure: {structure_break}")

        # 2. PREMIUM/DISCOUNT ANALYSIS
        pd_zone, key_levels = self._analyze_premium_discount(candles_current)
        notes.append(f"Price in {pd_zone}")

        # 3. ORDER FLOW ANALYSIS
        order_flow = self._analyze_order_flow(zones_current, candles_current)
        if order_flow != "NEUTRAL_OF":
            notes.append(f"Order Flow: {order_flow}")

        # 4. HTF BIAS (если есть данные старшего TF)
        htf_bias = None
        if candles_htf:
            htf_bias, _ = self._analyze_market_structure(candles_htf)
            notes.append(f"HTF Bias: {htf_bias}")

        # 5. РАСЧЕТ ФИНАЛЬНОГО BIAS
        final_bias, strength = self._calculate_final_bias(
            structure_bias=structure_bias,
            pd_zone=pd_zone,
            order_flow=order_flow,
            htf_bias=htf_bias,
            structure_break=structure_break
        )

        return BiasContext(
            bias=final_bias,
            htf_bias=htf_bias,
            structure_break=structure_break,
            premium_discount=pd_zone,
            order_flow=order_flow,
            strength=strength,
            key_levels=key_levels,
            notes=notes
        )

    def _analyze_market_structure(
            self,
            candles: List[Candle]
    ) -> Tuple[MarketBias, Optional[str]]:
        """
        Анализ структуры рынка с определением BOS/CHoCH
        """
        if len(candles) < 20:
            return MarketBias.NEUTRAL, None

        # Находим swing points
        swings = self._find_swing_points(candles)
        if len(swings) < 4:
            return MarketBias.NEUTRAL, None

        # Анализируем последние 4 swing points
        recent_swings = swings[-4:]

        # Определяем тренд по swing points
        highs = [s for s in recent_swings if s['type'] == 'HIGH']
        lows = [s for s in recent_swings if s['type'] == 'LOW']

        structure_break = None
        bias = MarketBias.NEUTRAL

        if len(highs) >= 2 and len(lows) >= 2:
            # Higher Highs & Higher Lows = BULLISH
            if highs[-1]['price'] > highs[-2]['price'] and lows[-1]['price'] > lows[-2]['price']:
                bias = MarketBias.BULLISH

                # Проверяем на BOS UP
                if self._check_bos_up(candles, highs[-2]['price']):
                    structure_break = "BOS_UP"
                    bias = MarketBias.STRONG_BULLISH

            # Lower Highs & Lower Lows = BEARISH
            elif highs[-1]['price'] < highs[-2]['price'] and lows[-1]['price'] < lows[-2]['price']:
                bias = MarketBias.BEARISH

                # Проверяем на BOS DOWN
                if self._check_bos_down(candles, lows[-2]['price']):
                    structure_break = "BOS_DOWN"
                    bias = MarketBias.STRONG_BEARISH

            # Mixed structure - проверяем CHoCH
            else:
                # CHoCH UP (смена с bearish на bullish)
                if highs[-1]['price'] > highs[-2]['price'] and self._was_bearish_before(swings):
                    structure_break = "CHoCH_UP"
                    bias = MarketBias.BULLISH

                # CHoCH DOWN (смена с bullish на bearish)
                elif lows[-1]['price'] < lows[-2]['price'] and self._was_bullish_before(swings):
                    structure_break = "CHoCH_DOWN"
                    bias = MarketBias.BEARISH

        return bias, structure_break

    def _analyze_premium_discount(
            self,
            candles: List[Candle]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Определяет, где находится цена относительно Premium/Discount
        """
        if len(candles) < 50:
            return "EQUILIBRIUM", {}

        # Находим range за последние N свечей
        lookback_candles = candles[-self.structure_lookback:]
        high = max(c.high for c in lookback_candles)
        low = min(c.low for c in lookback_candles)

        if high == low:
            return "EQUILIBRIUM", {}

        # Уровни Фибоначчи для ICT
        range_size = high - low
        equilibrium = low + (range_size * 0.5)
        premium_threshold = low + (range_size * 0.705)  # 70.5%
        discount_threshold = low + (range_size * 0.295)  # 29.5%
        optimal_trade_entry_premium = low + (range_size * 0.79)  # OTE 79%
        optimal_trade_entry_discount = low + (range_size * 0.21)  # OTE 21%

        current_price = candles[-1].close

        # Определяем зону
        if current_price >= premium_threshold:
            zone = "PREMIUM"
        elif current_price <= discount_threshold:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"

        key_levels = {
            "resistance": high,
            "support": low,
            "equilibrium": equilibrium,
            "premium_threshold": premium_threshold,
            "discount_threshold": discount_threshold,
            "ote_premium": optimal_trade_entry_premium,
            "ote_discount": optimal_trade_entry_discount
        }

        return zone, key_levels

    def _analyze_order_flow(
            self,
            zones: Optional[List[Zone]],
            candles: List[Candle]
    ) -> str:
        """
        Анализ Order Flow через зоны и объемы
        """
        if not zones or not candles:
            return "NEUTRAL_OF"

        current_price = candles[-1].close

        # Считаем bullish и bearish зоны
        bullish_zones = 0
        bearish_zones = 0

        # Анализируем зоны относительно текущей цены
        for zone in zones:
            zone_mid = (zone.high + zone.low) / 2

            # Bullish зоны
            if zone.type in ["OB_BULL", "FVG_UP", "BISI", "MB_BULL", "BREAKER_BULL"]:
                if zone_mid < current_price:  # Зона поддержки снизу
                    bullish_zones += 2
                else:
                    bullish_zones += 1

            # Bearish зоны
            elif zone.type in ["OB_BEAR", "FVG_DOWN", "SIBI", "MB_BEAR", "BREAKER_BEAR"]:
                if zone_mid > current_price:  # Зона сопротивления сверху
                    bearish_zones += 2
                else:
                    bearish_zones += 1

        # Анализ последних свечей для momentum
        recent_candles = candles[-10:]
        bullish_candles = sum(1 for c in recent_candles if c.close > c.open)
        bearish_candles = sum(1 for c in recent_candles if c.close < c.open)

        # Комбинированная оценка
        bullish_score = bullish_zones + bullish_candles
        bearish_score = bearish_zones + bearish_candles

        if bullish_score > bearish_score * 1.5:
            return "BULLISH_OF"
        elif bearish_score > bullish_score * 1.5:
            return "BEARISH_OF"
        else:
            return "NEUTRAL_OF"

    def _calculate_final_bias(
            self,
            structure_bias: MarketBias,
            pd_zone: str,
            order_flow: str,
            htf_bias: Optional[MarketBias],
            structure_break: Optional[str]
    ) -> Tuple[MarketBias, float]:
        """
        Расчет финального BIAS и его силы
        """
        score = 0
        max_score = 100

        # 1. Structure bias (30 points)
        structure_scores = {
            MarketBias.STRONG_BULLISH: 30,
            MarketBias.BULLISH: 20,
            MarketBias.NEUTRAL: 0,
            MarketBias.BEARISH: -20,
            MarketBias.STRONG_BEARISH: -30
        }
        score += structure_scores.get(structure_bias, 0)

        # 2. Premium/Discount (20 points)
        pd_scores = {
            "PREMIUM": -10 if structure_bias in [MarketBias.BULLISH, MarketBias.STRONG_BULLISH] else 10,
            "EQUILIBRIUM": 0,
            "DISCOUNT": 10 if structure_bias in [MarketBias.BULLISH, MarketBias.STRONG_BULLISH] else -10
        }
        score += pd_scores.get(pd_zone, 0)

        # 3. Order Flow (20 points)
        of_scores = {
            "BULLISH_OF": 20,
            "NEUTRAL_OF": 0,
            "BEARISH_OF": -20
        }
        score += of_scores.get(order_flow, 0)

        # 4. HTF Bias (20 points)
        if htf_bias:
            htf_scores = {
                MarketBias.STRONG_BULLISH: 20,
                MarketBias.BULLISH: 10,
                MarketBias.NEUTRAL: 0,
                MarketBias.BEARISH: -10,
                MarketBias.STRONG_BEARISH: -20
            }
            score += htf_scores.get(htf_bias, 0)

        # 5. Structure break bonus (10 points)
        if structure_break:
            if "UP" in structure_break:
                score += 10
            elif "DOWN" in structure_break:
                score -= 10

        # Нормализуем score в strength (0-100)
        strength = min(100, max(0, abs(score)))

        # Определяем финальный bias
        if score >= 40:
            return MarketBias.STRONG_BULLISH, strength
        elif score >= 20:
            return MarketBias.BULLISH, strength
        elif score <= -40:
            return MarketBias.STRONG_BEARISH, strength
        elif score <= -20:
            return MarketBias.BEARISH, strength
        else:
            return MarketBias.NEUTRAL, strength

    def _find_swing_points(self, candles: List[Candle]) -> List[Dict]:
        """
        Находит swing highs и lows
        """
        swings = []
        lookback = min(self.swing_lookback, len(candles) // 4)

        for i in range(lookback, len(candles) - lookback):
            c = candles[i]

            # Swing High
            is_swing_high = all(
                c.high >= candles[j].high
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )

            # Swing Low
            is_swing_low = all(
                c.low <= candles[j].low
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )

            if is_swing_high:
                swings.append({
                    'type': 'HIGH',
                    'price': c.high,
                    'index': i,
                    'time': c.time
                })

            if is_swing_low:
                swings.append({
                    'type': 'LOW',
                    'price': c.low,
                    'index': i,
                    'time': c.time
                })

        return sorted(swings, key=lambda x: x['index'])

    def _check_bos_up(self, candles: List[Candle], resistance_level: float) -> bool:
        """
        Проверяет Break of Structure вверх
        """
        # Последние 5 свечей
        recent = candles[-5:]

        # Был ли пробой и закрепление выше уровня
        for c in recent:
            if c.close > resistance_level * 1.001:  # Закрытие выше с запасом 0.1%
                return True
        return False

    def _check_bos_down(self, candles: List[Candle], support_level: float) -> bool:
        """
        Проверяет Break of Structure вниз
        """
        recent = candles[-5:]

        for c in recent:
            if c.close < support_level * 0.999:  # Закрытие ниже с запасом 0.1%
                return True
        return False

    def _was_bullish_before(self, swings: List[Dict]) -> bool:
        """
        Проверяет, был ли тренд bullish до последних swing points
        """
        if len(swings) < 6:
            return False

        older_swings = swings[-6:-4]
        highs = [s for s in older_swings if s['type'] == 'HIGH']
        lows = [s for s in older_swings if s['type'] == 'LOW']

        if len(highs) >= 2 and len(lows) >= 2:
            return highs[-1]['price'] > highs[-2]['price'] and lows[-1]['price'] > lows[-2]['price']
        return False

    def _was_bearish_before(self, swings: List[Dict]) -> bool:
        """
        Проверяет, был ли тренд bearish до последних swing points
        """
        if len(swings) < 6:
            return False

        older_swings = swings[-6:-4]
        highs = [s for s in older_swings if s['type'] == 'HIGH']
        lows = [s for s in older_swings if s['type'] == 'LOW']

        if len(highs) >= 2 and len(lows) >= 2:
            return highs[-1]['price'] < highs[-2]['price'] and lows[-1]['price'] < lows[-2]['price']
        return False