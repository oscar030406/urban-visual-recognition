from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_batch(value: str) -> int | float:
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained Ultralytics YOLO detector.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=parse_batch, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", default="outputs/val")
    parser.add_argument("--name", default=None)
    parser.add_argument("--aux-gate", choices=["none", "p3p4p5"], default="none")
    parser.add_argument("--aux-gate-alpha-init", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from ultralytics import YOLO
    from city_multimodal_detection.yolo_attention import attach_aux_pyramid_gates, register_ultralytics_attention

    register_ultralytics_attention()
    model = YOLO(str(args.weights))
    if args.aux_gate == "p3p4p5":
        gates = attach_aux_pyramid_gates(model.model, alpha=args.aux_gate_alpha_init)
        print(f"enabled AuxPyramidGates on layers {sorted(gates.layer_channels)}")
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        augment=args.augment,
        project=str(Path(args.project).resolve()),
        name=args.name,
    )
    print(f"mAP@50: {metrics.box.map50:.6f}")
    print(f"mAP@50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
