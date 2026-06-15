from pathlib import Path

from city_multimodal_detection.submission import Prediction, write_prediction_file


def test_write_prediction_file_clips_sorts_and_limits_boxes(tmp_path: Path):
    predictions = [
        Prediction(class_id=2, cx=1.2, cy=-0.1, w=0.3, h=0.4, confidence=0.4),
        Prediction(class_id=1, cx=0.5, cy=0.6, w=0.7, h=0.8, confidence=0.9),
        Prediction(class_id=99, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8),
    ]

    out_file = tmp_path / "000001.txt"
    write_prediction_file(out_file, predictions, max_det=1, num_classes=12)

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines == ["1 0.500000 0.600000 0.700000 0.800000 0.900000"]


def test_write_prediction_file_creates_empty_file_for_no_detections(tmp_path: Path):
    out_file = tmp_path / "empty.txt"

    write_prediction_file(out_file, [], max_det=100, num_classes=12)

    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == ""
