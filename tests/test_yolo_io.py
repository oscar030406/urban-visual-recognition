from pathlib import Path

import pytest

from city_multimodal_detection.yolo_io import (
    CLASS_NAMES,
    clip_yolo_label,
    make_data_yaml,
    parse_yolo_label_line,
    split_items,
)


def test_parse_yolo_label_line_accepts_five_values():
    parsed = parse_yolo_label_line("6 0.5 0.25 0.125 0.75")

    assert parsed == (6, 0.5, 0.25, 0.125, 0.75)


def test_parse_yolo_label_line_rejects_invalid_coordinate():
    try:
        parse_yolo_label_line("6 1.5 0.25 0.125 0.75")
    except ValueError as exc:
        assert "range" in str(exc)
    else:
        raise AssertionError("invalid coordinate was accepted")


def test_clip_yolo_label_clamps_box_to_image_bounds():
    parsed = parse_yolo_label_line("0 1.100000 0.500000 0.400000 0.200000", allow_out_of_bounds=True)

    clipped = clip_yolo_label(parsed)

    assert clipped == pytest.approx((0, 0.95, 0.5, 0.1, 0.2))


def test_split_items_is_deterministic():
    items = [f"item-{i}" for i in range(10)]

    train_a, val_a = split_items(items, val_ratio=0.2, seed=7)
    train_b, val_b = split_items(items, val_ratio=0.2, seed=7)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(val_a) == 2


def test_make_data_yaml_contains_detection_paths(tmp_path: Path):
    yaml_text = make_data_yaml(tmp_path, CLASS_NAMES)

    assert f"path: {tmp_path.as_posix()}" in yaml_text
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "person" in yaml_text
