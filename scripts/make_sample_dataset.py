from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python is required: python -m pip install opencv-python") from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.image_ops import write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny official-like dataset for smoke tests.")
    parser.add_argument("--out-root", type=Path, default=Path("tmp/sample_official_dataset"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dirname in ["RGB", "Infrared", "Depth", "labels"]:
        (args.out_root / dirname).mkdir(parents=True, exist_ok=True)

    for idx in range(args.count):
        sample_id = f"{idx:06d}"
        rgb = np.zeros((args.size, args.size, 3), dtype=np.uint8)
        infrared = np.zeros_like(rgb)
        depth = np.full((args.size, args.size), 5_000, dtype=np.uint16)

        x1 = 12 + idx * 3
        y1 = 16 + idx * 2
        x2 = min(args.size - 8, x1 + 28)
        y2 = min(args.size - 8, y1 + 24)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (40, 180, 240), -1)
        cv2.rectangle(infrared, (x1, y1), (x2, y2), (180, 180, 180), -1)
        depth[y1:y2, x1:x2] = 1_200

        write_image(args.out_root / "RGB" / f"{sample_id}.png", rgb)
        write_image(args.out_root / "Infrared" / f"{sample_id}.png", infrared)
        write_image(args.out_root / "Depth" / f"{sample_id}.png", depth)

        cx = ((x1 + x2) / 2) / args.size
        cy = ((y1 + y2) / 2) / args.size
        width = (x2 - x1) / args.size
        height = (y2 - y1) / args.size
        (args.out_root / "labels" / f"{sample_id}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n",
            encoding="utf-8",
        )

    print(args.out_root)


if __name__ == "__main__":
    main()
