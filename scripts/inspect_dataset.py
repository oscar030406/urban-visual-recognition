from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.dataset import discover_records, infer_dataset_dirs
from city_multimodal_detection.yolo_io import CLASS_NAMES, read_label_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect official RGB/IR/Depth detection data.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--allow-missing-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = infer_dataset_dirs(args.raw_root)
    records = discover_records(args.raw_root, require_labels=not args.allow_missing_labels)
    class_counts: collections.Counter[int] = collections.Counter()
    box_count = 0
    for record in records:
        if record.label is None:
            continue
        labels = read_label_file(record.label, clip_boxes=True)
        box_count += len(labels)
        class_counts.update(label[0] for label in labels)

    print(f"RGB: {dirs.rgb}")
    print(f"Infrared: {dirs.infrared}")
    print(f"Depth: {dirs.depth}")
    print(f"Labels: {dirs.labels}")
    print(f"paired samples: {len(records)}")
    print(f"boxes: {box_count}")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{class_id:02d} {name}: {class_counts[class_id]}")


if __name__ == "__main__":
    main()
