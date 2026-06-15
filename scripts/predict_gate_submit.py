from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.dataset import discover_records
from city_multimodal_detection.image_ops import make_rgb_ir_depth_gate_array, read_image
from city_multimodal_detection.submission import Prediction, write_prediction_file
from city_multimodal_detection.yolo_io import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB-main IR/depth-gated YOLO inference and create official TXT zip."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True, help="Official test root with visible/infrared/depth.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=0)
    return parser.parse_args()


def zip_submission(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)


def preprocess(array: np.ndarray, imgsz: int, stride: int) -> tuple[torch.Tensor, tuple[int, int]]:
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True, stride=stride)
    padded = letterbox(image=array)
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.float() / 255.0, padded.shape[:2]


def predictions_from_tensor(
    model,
    tensor: torch.Tensor,
    padded_shape: tuple[int, int],
    original_shape: tuple[int, int],
    conf: float,
    iou: float,
    max_det: int,
) -> list[Prediction]:
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, xyxy2xywh

    raw = model(tensor)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    detections = non_max_suppression(
        raw,
        conf_thres=conf,
        iou_thres=iou,
        max_det=max_det,
        nc=len(CLASS_NAMES),
    )[0]
    if detections is None or len(detections) == 0:
        return []

    detections[:, :4] = scale_boxes(padded_shape, detections[:, :4], original_shape).clamp_(0)
    xywh = xyxy2xywh(detections[:, :4])
    height, width = original_shape
    predictions: list[Prediction] = []
    for box, row in zip(xywh, detections):
        cx, cy, bw, bh = box.tolist()
        predictions.append(
            Prediction(
                class_id=int(row[5].item()),
                cx=float(cx / width),
                cy=float(cy / height),
                w=float(bw / width),
                h=float(bh / height),
                confidence=float(row[4].item()),
            )
        )
    return predictions


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    from city_multimodal_detection.yolo_attention import register_ultralytics_attention
    from ultralytics import YOLO

    args = parse_args()
    register_ultralytics_attention()
    records = discover_records(args.raw_root, require_labels=False)
    if not records:
        raise SystemExit(f"no usable RGB/IR/depth records found under {args.raw_root}")

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.weights)).model.to(device).eval()
    stride = max(int(model.stride.max().item()) if hasattr(model, "stride") else 32, 32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_prediction_file(args.out_dir / f"{record.sample_id}.txt", [], max_det=args.max_det)

    with torch.inference_mode():
        for record in records:
            rgb = read_image(record.rgb)
            infrared = read_image(record.infrared)
            depth = read_image(record.depth, unchanged=True)
            array = make_rgb_ir_depth_gate_array(rgb, infrared, depth)
            tensor, padded_shape = preprocess(array, imgsz=args.imgsz, stride=stride)
            tensor = tensor.to(device, non_blocking=True)
            predictions = predictions_from_tensor(
                model=model,
                tensor=tensor,
                padded_shape=padded_shape,
                original_shape=array.shape[:2],
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
            )
            write_prediction_file(args.out_dir / f"{record.sample_id}.txt", predictions, max_det=args.max_det)

    if args.zip_path:
        zip_submission(args.out_dir, args.zip_path)
        print(args.zip_path)
    else:
        print(args.out_dir)


if __name__ == "__main__":
    main()
