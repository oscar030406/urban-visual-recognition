from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    truthy = {"1", "true", "yes", "y", "tta", "augment"}
    falsy = {"0", "false", "no", "n", "none", "noaugment"}
    parsed: list[bool] = []
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in truthy:
            parsed.append(True)
        elif token in falsy:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid bool value: {item}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Ultralytics YOLO validation inference parameters.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=parse_int_list, default=[1280])
    parser.add_argument("--conf", type=parse_float_list, default=[0.001])
    parser.add_argument("--iou", type=parse_float_list, default=[0.7])
    parser.add_argument("--augment", type=parse_bool_list, default=[False])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default="outputs/val_sweeps")
    parser.add_argument("--name-prefix", default="sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from ultralytics import YOLO

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    model = YOLO(str(args.weights))

    with args.output.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "time",
                "weights",
                "data",
                "imgsz",
                "conf",
                "iou",
                "augment",
                "batch",
                "device",
                "map50",
                "map50_95",
            ],
        )
        if not exists:
            writer.writeheader()

        for imgsz in args.imgsz:
            for conf in args.conf:
                for iou in args.iou:
                    for augment in args.augment:
                        name = f"{args.name_prefix}_i{imgsz}_c{conf:g}_n{iou:g}_tta{int(augment)}"
                        print(
                            "sweep "
                            f"imgsz={imgsz} conf={conf:g} iou={iou:g} augment={augment} batch={args.batch}"
                        )
                        metrics = model.val(
                            data=str(args.data),
                            imgsz=imgsz,
                            conf=conf,
                            iou=iou,
                            augment=augment,
                            batch=args.batch,
                            device=args.device,
                            project=str(Path(args.project).resolve()),
                            name=name,
                            verbose=False,
                        )
                        row = {
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "weights": str(args.weights),
                            "data": str(args.data),
                            "imgsz": imgsz,
                            "conf": conf,
                            "iou": iou,
                            "augment": augment,
                            "batch": args.batch,
                            "device": args.device,
                            "map50": f"{metrics.box.map50:.6f}",
                            "map50_95": f"{metrics.box.map:.6f}",
                        }
                        writer.writerow(row)
                        fp.flush()
                        print(f"result mAP50={row['map50']} mAP50-95={row['map50_95']}")


if __name__ == "__main__":
    main()
