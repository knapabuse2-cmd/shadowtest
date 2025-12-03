from dataclasses import dataclass
from typing import List, Optional, Callable, Dict


@dataclass
class Zone:
    tf: str
    high: float
    low: float
    type: str  # OB_BULL / OB_BEAR / FVG_UP / FVG_DOWN / FH / FL
    timestamp: Optional[int] = None  # ДОБАВЛЕНО: время появления зоны
    candle_index: Optional[int] = None  # ДОБАВЛЕНО: индекс свечи


@dataclass
class VolumeContext:
    tf: str
    bias: str
    structure: str
    note: str
    killzone: str = "None"


@dataclass
class DetectionResult:
    zones: List[Zone]
    context: Optional[VolumeContext]


@dataclass
class ChainSignal:
    symbol: str
    chain_id: str
    tf: str
    direction: str
    entry: float
    stop_loss: float
    take_profits: List[float]
    rr: float
    description: str
    timestamp: Optional[int] = None  # ДОБАВЛЕНО: время сигнала


# ИСПРАВЛЕНО: Добавляем импорт BiasContext
from dataclasses import field

@dataclass
class BiasContext:
    """Контекст рыночного bias для фильтрации сигналов"""
    bias: str  # STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
    htf_bias: Optional[str] = None  # Bias старшего TF
    structure_break: Optional[str] = None  # BOS_UP, BOS_DOWN, CHoCH_UP, CHoCH_DOWN
    premium_discount: Optional[str] = None  # PREMIUM, EQUILIBRIUM, DISCOUNT
    order_flow: Optional[str] = None  # BULLISH_OF, BEARISH_OF, NEUTRAL_OF
    strength: float = 0.0  # 0-100
    key_levels: Dict[str, float] = field(default_factory=dict)  # resistance, support, equilibrium
    notes: List[str] = field(default_factory=list)


@dataclass
class ChainContext:
    symbol: str
    candles: dict
    detections: dict
    bias_contexts: Optional[Dict[str, BiasContext]] = None  # ИСПРАВЛЕНО: добавлено поле
    log_callback: Optional[Callable] = None

    async def log(self, msg: str):
        full_msg = f"[{self.symbol}] {msg}"
        if self.log_callback:
            await self.log_callback(full_msg)
        else:
            print(full_msg)