from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.constants import IMAGE_EXTENSIONS
from city_multimodal_detection.dataset import FUSION_CHOICES, prepare_yolo_dataset
from city_multimodal_detection.submission import Prediction, write_prediction_file
from city_multimodal_detection.ultralytics_adapter import result_to_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference and generate a submission with per-class confidence thresholds."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/inference_inputs"))
    parser.add_argument("--fusion", choices=FUSION_CHOICES, default="rgb")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--class-conf", required=True, help="JSON file, 12 comma values, or class:threshold pairs.")
    parser.add_argument("--imgsz", type=int, default=1408)
    parser.add_argument("--base-conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--candidate-max-det", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--device", default=0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def parse_class_conf(value: str, num_classes: int) -> list[float]:
    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "class_conf" in data:
            data = data["class_conf"]
        if isinstance(data, dict):
            thresholds = [0.0] * num_classes
            for key, threshold in data.items():
                thresholds[int(key)] = float(threshold)
            return thresholds
        if isinstance(data, list):
            if len(data) != num_classes:
                raise ValueError(f"expected {num_classes} class thresholds, got {len(data)}")
            return [float(item) for item in data]
        raise ValueError(f"unsupported JSON threshold shape in {path}")

    if ":" in value:
        thresholds = [0.0] * num_classes
        for item in value.split(","):
            if not item.strip():
                continue
            key, threshold = item.split(":", maxsplit=1)
            thresholds[int(key.strip())] = float(threshold)
        return thresholds

    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(thresholds) != num_classes:
        raise ValueError(f"expected {num_classes} comma thresholds, got {len(thresholds)}")
    return thresholds


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def zip_submission(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)


def filter_predictions(predictions: list[Prediction], class_conf: list[float]) -> list[Prediction]:
    filtered = []
    for pred in predictions:
        class_id = int(pred.class_id)
        if 0 <= class_id < len(class_conf) and float(pred.confidence) >= class_conf[class_id]:
            filtered.append(pred)
    return filtered


def main() -> None:
    args = parse_args()
    class_conf = parse_class_conf(args.class_conf, args.num_classes)
    base_conf = min(class_conf) if args.base_conf is None else args.base_conf

    if args.image_dir is None:
        if args.raw_root is None:
            raise SystemExit("either --image-dir or --raw-root is required")
        prepared_root = args.work_dir / args.fusion
        if prepared_root.exists():
            shutil.rmtree(prepared_root)
        prepare_yolo_dataset(
            raw_root=args.raw_root,
            output_root=prepared_root,
            fusion=args.fusion,
            require_labels=False,
        )
        image_dir = prepared_root / "images" / "test"
    else:
        image_dir = args.image_dir

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"no images found under {image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        write_prediction_file(args.out_dir / f"{image.stem}.txt", [], max_det=args.max_det)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    total_before = 0
    total_after = 0
    for result in model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=base_conf,
        iou=args.iou,
        max_det=args.candidate_max_det,
        device=args.device,
        augment=args.augment,
        stream=True,
        save=False,
        verbose=False,
    ):
        predictions = result_to_predictions(result)
        total_before += len(predictions)
        filtered = filter_predictions(predictions, class_conf)
        total_after += len(filtered)
        write_prediction_file(args.out_dir / f"{Path(result.path).stem}.txt", filtered, max_det=args.max_det)

    zip_submission(args.out_dir, args.zip_path)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "weights": str(args.weights),
                    "fusion": args.fusion,
                    "imgsz": args.imgsz,
                    "base_conf": base_conf,
                    "class_conf": class_conf,
                    "iou": args.iou,
                    "candidate_max_det": args.candidate_max_det,
                    "max_det": args.max_det,
                    "augment": args.augment,
                    "images": len(images),
                    "predictions_before_class_filter": total_before,
                    "predictions_after_class_filter": total_after,
                    "zip_path": str(args.zip_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(args.zip_path)
    print(f"predictions before class filter: {total_before}")
    print(f"predictions after class filter: {total_after}")


if __name__ == "__main__":
    main()

