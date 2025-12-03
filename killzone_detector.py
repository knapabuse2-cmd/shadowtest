from analysis_interfaces import DetectionResult, VolumeContext


class KillzoneDetector:
    """
    Упрощённый детектор Killzone.
    Сейчас killzone = "None", чтобы не ломать цепочки.
    Позже можно сделать:
        - London (08:00–11:00 UTC)
        - NY AM (13:00–16:00 UTC)
        - NY PM (18:00–20:00 UTC)
    """

    def detect(self, candles, tf: str):
        # Минимальная проверка
        if candles is None or len(candles) == 0:
            return DetectionResult([], None)

        # Пока просто ставим killzone=None
        context = VolumeContext(
            tf=tf,
            bias="Neutral",
            structure="",
            note="killzone=None",
            killzone="None"
        )

        return DetectionResult([], context)
