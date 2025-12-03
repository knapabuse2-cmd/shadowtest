# volume_profile_detector.py
# ИСПРАВЛЕНО: Добавлены все необходимые импорты

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from data.data_interfaces import Candle


@dataclass
class VolumeProfile:
    poc: float  # Point of Control - уровень с макс. объемом
    vah: float  # Value Area High
    val: float  # Value Area Low
    hvn: List[float]  # High Volume Nodes
    lvn: List[float]  # Low Volume Nodes


class VolumeProfileDetector:
    """
    Определяет ключевые уровни по объему
    """

    def detect(self, candles: List[Candle], tf: str) -> VolumeProfile:
        """
        Строит профиль объема и возвращает ключевые уровни.
        """
        if not candles or len(candles) < 10:
            # Возвращаем пустой профиль
            return VolumeProfile(
                poc=0.0,
                vah=0.0,
                val=0.0,
                hvn=[],
                lvn=[]
            )

        # Построение профиля объема
        price_levels: Dict[float, float] = {}

        for candle in candles[-100:]:  # Последние 100 свечей
            # Разбиваем свечу на уровни
            levels = np.linspace(candle.low, candle.high, 10)
            volume_per_level = candle.volume / len(levels)

            for level in levels:
                rounded = round(level, 2)
                price_levels[rounded] = price_levels.get(rounded, 0) + volume_per_level

        if not price_levels:
            return VolumeProfile(
                poc=0.0,
                vah=0.0,
                val=0.0,
                hvn=[],
                lvn=[]
            )

        # Находим POC (Point of Control)
        poc = max(price_levels, key=price_levels.get)

        # Value Area (70% объема)
        sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(price_levels.values())
        value_area_volume = 0.0
        value_area_levels: List[float] = []

        for price, vol in sorted_levels:
            value_area_volume += vol
            value_area_levels.append(price)
            if value_area_volume >= total_volume * 0.7:
                break

        vah = max(value_area_levels) if value_area_levels else poc
        val = min(value_area_levels) if value_area_levels else poc

        # HVN/LVN (узлы высокого/низкого объема)
        avg_volume = total_volume / len(price_levels) if price_levels else 0
        hvn = [p for p, v in price_levels.items() if v > avg_volume * 1.5]
        lvn = [p for p, v in price_levels.items() if v < avg_volume * 0.5]

        return VolumeProfile(poc=poc, vah=vah, val=val, hvn=hvn, lvn=lvn)
