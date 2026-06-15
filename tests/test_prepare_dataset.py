from pathlib import Path

import numpy as np
import pytest

from city_multimodal_detection.dataset import prepare_yolo_dataset


cv2 = pytest.importorskip("cv2")


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)


def _make_raw_dataset(raw: Path, count: int = 3) -> None:
    for dirname in ["RGB", "Infrared", "Depth", "labels"]:
        (raw / dirname).mkdir(parents=True)

    for idx in range(count):
        sample_id = f"{idx:06d}"
        _write_png(raw / "RGB" / f"{sample_id}.png", np.full((8, 8, 3), idx * 30 + 20, np.uint8))
        _write_png(raw / "Infrared" / f"{sample_id}.png", np.full((8, 8, 3), 80, np.uint8))
        _write_png(raw / "Depth" / f"{sample_id}.png", np.full((8, 8), 1_000, np.uint16))
        (raw / "labels" / f"{sample_id}.txt").write_text(
            "0 0.500000 0.500000 0.250000 0.250000\n",
            encoding="utf-8",
        )


def test_prepare_yolo_dataset_materializes_triad3(tmp_path: Path):
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)

    data_yaml = prepare_yolo_dataset(raw, tmp_path / "prepared", fusion="triad3", val_ratio=0.34)

    assert data_yaml.exists()
    assert (tmp_path / "prepared" / "records.jsonl").exists()
    assert len(list((tmp_path / "prepared" / "images").rglob("*.png"))) == 3
    assert len(list((tmp_path / "prepared" / "labels").rglob("*.txt"))) == 3


def test_prepare_yolo_dataset_materializes_rgb_guided_rdt(tmp_path: Path):
    raw = tmp_path / "raw"
    _make_raw_dataset(raw)

    data_yaml = prepare_yolo_dataset(raw, tmp_path / "prepared_rdt", fusion="rgb_guided_rdt", val_ratio=0.34)

    assert data_yaml.exists()
    assert len(list((tmp_path / "prepared_rdt" / "images").rglob("*.png"))) == 3
    materialized = cv2.imdecode(
        np.fromfile(next((tmp_path / "prepared_rdt" / "images").rglob("*.png")), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert materialized is not None
    assert materialized.shape == (8, 8, 3)


def test_prepare_yolo_dataset_can_create_modality_dropout_variants(tmp_path: Path):
    raw = tmp_path / "raw"
    _make_raw_dataset(raw, count=2)

    data_yaml = prepare_yolo_dataset(
        raw,
        tmp_path / "prepared_dropout",
        fusion="cssa3",
        val_ratio=0.5,
        modality_dropout=True,
    )

    assert data_yaml.exists()
    train_images = list((tmp_path / "prepared_dropout" / "images" / "train").glob("*.png"))
    train_labels = list((tmp_path / "prepared_dropout" / "labels" / "train").glob("*.txt"))
    assert any("_drop_ir" in path.stem for path in train_images)
    assert any("_drop_depth" in path.stem for path in train_images)
    assert len(train_images) == len(train_labels)


def test_prepare_yolo_dataset_clips_off_image_labels(tmp_path: Path):
    raw = tmp_path / "raw"
    _make_raw_dataset(raw, count=2)
    (raw / "labels" / "000000.txt").write_text(
        "0 1.100000 0.500000 0.400000 0.200000\n",
        encoding="utf-8",
    )

    prepare_yolo_dataset(raw, tmp_path / "prepared_clipped", fusion="rgb", val_ratio=0.5)

    label_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "prepared_clipped" / "labels").rglob("000000.txt"))
    )
    assert "0 0.950000 0.500000 0.100000 0.200000" in label_text
