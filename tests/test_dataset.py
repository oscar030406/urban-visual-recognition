from pathlib import Path

from city_multimodal_detection.dataset import discover_records, infer_dataset_dirs


def test_infer_dataset_dirs_finds_common_modality_names(tmp_path: Path):
    (tmp_path / "RGB").mkdir()
    (tmp_path / "Infrared").mkdir()
    (tmp_path / "Depth").mkdir()
    (tmp_path / "labels").mkdir()

    dirs = infer_dataset_dirs(tmp_path)

    assert dirs.rgb == tmp_path / "RGB"
    assert dirs.infrared == tmp_path / "Infrared"
    assert dirs.depth == tmp_path / "Depth"
    assert dirs.labels == tmp_path / "labels"


def test_discover_records_pairs_modalities_and_labels_by_stem(tmp_path: Path):
    for dirname in ["RGB", "Infrared", "Depth", "labels"]:
        (tmp_path / dirname).mkdir()
    for dirname, suffix in [("RGB", ".png"), ("Infrared", ".png"), ("Depth", ".png")]:
        (tmp_path / dirname / f"000001{suffix}").write_bytes(b"placeholder")
    (tmp_path / "labels" / "000001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    records = discover_records(tmp_path, require_labels=True)

    assert len(records) == 1
    assert records[0].sample_id == "000001"
    assert records[0].rgb.name == "000001.png"
    assert records[0].label is not None
