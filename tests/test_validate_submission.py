from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_submission_script", ROOT / "scripts" / "validate_submission.py"
)
assert SPEC and SPEC.loader
validate_submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_submission)


def test_validate_submission_accepts_empty_txt_files(tmp_path: Path):
    out_dir = tmp_path / "submission"
    out_dir.mkdir()
    (out_dir / "000001.txt").write_text("", encoding="utf-8")
    (out_dir / "000002.txt").write_text(
        "1 0.500000 0.500000 0.100000 0.200000 0.900000\n",
        encoding="utf-8",
    )

    errors = validate_submission.validate_submission(out_dir, {"000001", "000002"})

    assert errors == []


def test_validate_submission_reports_missing_and_bad_rows(tmp_path: Path):
    out_dir = tmp_path / "submission"
    out_dir.mkdir()
    (out_dir / "000001.txt").write_text("99 1.2 0.5 0.1 0.2 0.9\n", encoding="utf-8")

    errors = validate_submission.validate_submission(out_dir, {"000001", "000002"})

    assert any("missing txt files" in error for error in errors)
    assert any("class_id out of range" in error for error in errors)
    assert any("cx out of range" in error for error in errors)


def test_validate_submission_reads_flat_zip(tmp_path: Path):
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("000001.txt", "")

    errors = validate_submission.validate_submission(zip_path, {"000001"})

    assert errors == []


def test_validate_submission_rejects_nested_zip_layout(tmp_path: Path):
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nested/000001.txt", "")

    errors = validate_submission.validate_submission(zip_path, {"000001"})

    assert any("flat TXT files" in error for error in errors)
