from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.dataset import FUSION_CHOICES, prepare_yolo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official RGB/IR/Depth data for YOLO.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Official dataset root.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output YOLO dataset root.")
    parser.add_argument("--fusion", choices=FUSION_CHOICES, default="rgb")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-strategy", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Parallel workers for image materialization.",
    )
    parser.add_argument("--test-only", action="store_true", help="Prepare images without labels.")
    parser.add_argument(
        "--modality-dropout",
        action="store_true",
        help="For fused train images, add drop_rgb/drop_ir/drop_depth variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = prepare_yolo_dataset(
        raw_root=args.raw_root,
        output_root=args.out_root,
        fusion=args.fusion,
        val_ratio=args.val_ratio,
        seed=args.seed,
        link_strategy=args.link_strategy,
        require_labels=not args.test_only,
        workers=args.workers,
        modality_dropout=args.modality_dropout,
    )
    print(output)


if __name__ == "__main__":
    main()
