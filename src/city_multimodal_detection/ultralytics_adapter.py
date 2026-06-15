from __future__ import annotations

from typing import Any

import numpy as np

from .submission import Prediction


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def result_to_predictions(result: Any) -> list[Prediction]:
    """Convert one Ultralytics Result object into official normalized predictions."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xywhn = _to_numpy(getattr(boxes, "xywhn", []))
    classes = _to_numpy(getattr(boxes, "cls", []))
    confidences = _to_numpy(getattr(boxes, "conf", []))

    predictions: list[Prediction] = []
    for xywh, class_id, confidence in zip(xywhn, classes, confidences):
        cx, cy, width, height = [float(value) for value in xywh[:4]]
        predictions.append(
            Prediction(
                class_id=int(class_id),
                cx=cx,
                cy=cy,
                w=width,
                h=height,
                confidence=float(confidence),
            )
        )
    return predictions
