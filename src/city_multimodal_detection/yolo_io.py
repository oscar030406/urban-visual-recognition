from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TypeVar

from .constants import CLASS_NAMES

T = TypeVar("T")
YoloLabel = tuple[int, float, float, float, float]


def parse_yolo_label_line(
    line: str,
    num_classes: int = len(CLASS_NAMES),
    allow_out_of_bounds: bool = False,
) -> YoloLabel:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 YOLO label values, got {len(parts)}: {line!r}")

    class_id = int(float(parts[0]))
    if class_id < 0 or class_id >= num_classes:
        raise ValueError(f"class_id out of range: {class_id}")

    coords = tuple(float(value) for value in parts[1:])
    if any(not math.isfinite(value) for value in coords):
        raise ValueError(f"YOLO coordinate must be finite: {line!r}")
    if coords[2] <= 0.0 or coords[3] <= 0.0:
        raise ValueError(f"YOLO width/height must be positive: {line!r}")
    if not allow_out_of_bounds and any(value < 0.0 or value > 1.0 for value in coords):
        raise ValueError(f"YOLO coordinate out of range 0..1: {line!r}")
    return (class_id, *coords)


def clip_yolo_label(label: YoloLabel) -> YoloLabel | None:
    class_id, center_x, center_y, width, height = label
    x1 = max(0.0, center_x - width / 2.0)
    y1 = max(0.0, center_y - height / 2.0)
    x2 = min(1.0, center_x + width / 2.0)
    y2 = min(1.0, center_y + height / 2.0)
    if x2 <= x1 or y2 <= y1:
        return None
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    return (
        class_id,
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        clipped_width,
        clipped_height,
    )


def format_yolo_label(label: YoloLabel) -> str:
    class_id, center_x, center_y, width, height = label
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def read_label_file(
    path: Path,
    num_classes: int = len(CLASS_NAMES),
    clip_boxes: bool = False,
) -> list[YoloLabel]:
    parsed = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = parse_yolo_label_line(
                line,
                num_classes=num_classes,
                allow_out_of_bounds=clip_boxes,
            )
            if clip_boxes:
                label = clip_yolo_label(label)
                if label is None:
                    continue
            parsed.append(label)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return parsed


def validate_label_file(path: Path, num_classes: int = len(CLASS_NAMES)) -> list[YoloLabel]:
    return read_label_file(path, num_classes=num_classes, clip_boxes=False)


def write_label_file(path: Path, labels: list[YoloLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(format_yolo_label(label) for label in labels)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def sanitize_label_file(source: Path, target: Path, num_classes: int = len(CLASS_NAMES)) -> list[YoloLabel]:
    labels = read_label_file(source, num_classes=num_classes, clip_boxes=True)
    write_label_file(target, labels)
    return labels


def split_items(items: list[T], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[T], list[T]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1, inclusive of 0")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if val_ratio == 0.0:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if shuffled else 0
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


def make_data_yaml(
    root: Path,
    class_names: list[str] | tuple[str, ...] = CLASS_NAMES,
    val_path: str = "images/val",
    channels: int = 3,
) -> str:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    return (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        f"val: {val_path}\n"
        f"channels: {channels}\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names}\n"
    )
