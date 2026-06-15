from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.constants import IMAGE_EXTENSIONS
from city_multimodal_detection.dataset import discover_records


def list_image_stems(image_dir: Path) -> set[str]:
    return {
        path.stem
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def expected_stems(raw_root: Path | None, image_dir: Path | None) -> set[str]:
    if raw_root is None and image_dir is None:
        raise ValueError("either raw_root or image_dir is required")
    if image_dir is not None:
        stems = list_image_stems(image_dir)
    else:
        assert raw_root is not None
        stems = {record.sample_id for record in discover_records(raw_root, require_labels=False)}
    if not stems:
        raise ValueError("no expected test images found")
    return stems


def read_submission_files(submission: Path) -> dict[str, str]:
    if submission.is_dir():
        return {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(submission.glob("*.txt"))
            if path.is_file()
        }
    if submission.suffix.lower() != ".zip":
        raise ValueError("submission must be a directory or .zip file")
    with zipfile.ZipFile(submission) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        files = {}
        for name in names:
            path = Path(name)
            if path.suffix.lower() == ".txt":
                files[path.name] = zf.read(name).decode("utf-8")
        return files


def validate_submission_layout(submission: Path) -> list[str]:
    errors: list[str] = []
    if submission.is_dir():
        nested = sorted(
            str(path.relative_to(submission))
            for path in submission.rglob("*.txt")
            if path.is_file() and path.parent != submission
        )
        if nested:
            errors.append(f"submission directory must be flat; nested txt first={nested[:5]}")
        return errors

    if submission.suffix.lower() != ".zip":
        return ["submission must be a directory or .zip file"]

    with zipfile.ZipFile(submission) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
    nested = [name for name in names if Path(name).name != name]
    non_txt = [name for name in names if Path(name).suffix.lower() != ".txt"]
    if nested:
        errors.append(f"zip must contain flat TXT files only; nested first={nested[:5]}")
    if non_txt:
        errors.append(f"zip contains non-TXT files; first={non_txt[:5]}")
    return errors


def validate_prediction_text(
    text: str,
    max_det: int = 100,
    num_classes: int = 12,
) -> list[str]:
    errors: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > max_det:
        errors.append(f"too many detections: {len(lines)} > {max_det}")
    for index, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 6:
            errors.append(f"line {index}: expected 6 columns, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(part) for part in parts[1:]]
        except ValueError:
            errors.append(f"line {index}: non-numeric value")
            continue
        if not 0 <= class_id < num_classes:
            errors.append(f"line {index}: class_id out of range: {class_id}")
        for name, value in zip(("cx", "cy", "w", "h", "confidence"), values, strict=True):
            if not 0.0 <= value <= 1.0:
                errors.append(f"line {index}: {name} out of range: {value}")
    return errors


def validate_submission(
    submission: Path,
    expected: set[str],
    max_det: int = 100,
    num_classes: int = 12,
) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_submission_layout(submission))
    files = read_submission_files(submission)
    expected_txt = {f"{stem}.txt" for stem in expected}
    actual_txt = set(files)

    missing = sorted(expected_txt - actual_txt)
    extra = sorted(actual_txt - expected_txt)
    if missing:
        errors.append(f"missing txt files: {len(missing)}; first={missing[:5]}")
    if extra:
        errors.append(f"extra txt files: {len(extra)}; first={extra[:5]}")

    for name in sorted(expected_txt & actual_txt):
        line_errors = validate_prediction_text(files[name], max_det=max_det, num_classes=num_classes)
        errors.extend(f"{name}: {error}" for error in line_errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate official TXT-per-image submission files.")
    parser.add_argument("--submission", type=Path, required=True, help="Submission directory or zip file.")
    parser.add_argument("--raw-root", type=Path, help="Official test root with RGB/IR/Depth folders.")
    parser.add_argument("--image-dir", type=Path, help="Prepared test image directory.")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = expected_stems(args.raw_root, args.image_dir)
    errors = validate_submission(
        submission=args.submission,
        expected=expected,
        max_det=args.max_det,
        num_classes=args.num_classes,
    )
    if errors:
        print(f"submission invalid: {len(errors)} error(s)")
        for error in errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"submission valid: {len(expected)} TXT files")


if __name__ == "__main__":
    main()
