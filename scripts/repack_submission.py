from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.dataset import discover_records
from validate_submission import read_submission_files, validate_prediction_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack a prediction directory/zip into a strict flat TXT zip ordered by official test samples."
    )
    parser.add_argument("--submission", type=Path, required=True, help="Existing prediction directory or zip.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Official test root with RGB/IR/Depth folders.")
    parser.add_argument("--out-zip", type=Path, required=True, help="Output flat zip path.")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = discover_records(args.raw_root, require_labels=False)
    expected_ids = [record.sample_id for record in records]
    if not expected_ids:
        raise SystemExit(f"no test records found under {args.raw_root}")

    files = read_submission_files(args.submission)
    expected_txt = {f"{sample_id}.txt" for sample_id in expected_ids}
    actual_txt = set(files)
    missing = sorted(expected_txt - actual_txt)
    extra = sorted(actual_txt - expected_txt)
    if missing or extra:
        if missing:
            print(f"missing txt files: {len(missing)}; first={missing[:10]}")
        if extra:
            print(f"extra txt files: {len(extra)}; first={extra[:10]}")
        raise SystemExit(1)

    errors: list[str] = []
    for sample_id in expected_ids:
        name = f"{sample_id}.txt"
        errors.extend(
            f"{name}: {error}"
            for error in validate_prediction_text(
                files[name],
                max_det=args.max_det,
                num_classes=args.num_classes,
            )
        )
    if errors:
        print(f"invalid prediction rows: {len(errors)}")
        for error in errors[:50]:
            print(f"- {error}")
        raise SystemExit(1)

    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sample_id in expected_ids:
            name = f"{sample_id}.txt"
            zf.writestr(name, files[name])

    print(f"wrote {args.out_zip}")
    print(f"txt_count={len(expected_ids)}")
    print("layout=flat")


if __name__ == "__main__":
    main()
