from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def write_prediction_file(
    path: Path,
    predictions: list[Prediction],
    max_det: int = 100,
    num_classes: int = 12,
) -> None:
    """Write one official prediction TXT file, creating an empty file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_predictions = [
        pred
        for pred in predictions
        if 0 <= int(pred.class_id) < num_classes and 0.0 <= float(pred.confidence) <= 1.0
    ]
    valid_predictions.sort(key=lambda pred: pred.confidence, reverse=True)
    lines = []
    for pred in valid_predictions[:max_det]:
        lines.append(
            f"{int(pred.class_id)} "
            f"{_clip01(pred.cx):.6f} "
            f"{_clip01(pred.cy):.6f} "
            f"{_clip01(pred.w):.6f} "
            f"{_clip01(pred.h):.6f} "
            f"{_clip01(pred.confidence):.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
