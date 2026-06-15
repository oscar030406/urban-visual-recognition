from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_yolo_script", ROOT / "scripts" / "train_yolo.py")
assert SPEC and SPEC.loader
train_yolo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_yolo)


def test_parse_batch_accepts_auto_free():
    assert train_yolo.parse_batch("auto-free") == "auto-free"
    assert train_yolo.parse_batch("auto_remaining") == "auto-free"


def test_auto_free_batch_fraction_uses_remaining_memory_with_reserve():
    fraction = train_yolo.auto_free_batch_fraction(
        total_mb=24_000,
        used_mb=4_000,
        reserve_mb=1_000,
        max_fraction=0.90,
    )

    assert fraction == pytest.approx(0.792)


def test_auto_free_batch_fraction_caps_when_gpu_is_empty():
    fraction = train_yolo.auto_free_batch_fraction(
        total_mb=24_000,
        used_mb=100,
        reserve_mb=1_000,
        max_fraction=0.90,
    )

    assert fraction == 0.90
