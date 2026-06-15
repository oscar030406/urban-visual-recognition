from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.constants import IMAGE_EXTENSIONS
from city_multimodal_detection.dataset import FUSION_CHOICES, prepare_yolo_dataset
from city_multimodal_detection.submission import write_prediction_file
from city_multimodal_detection.ultralytics_adapter import result_to_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and generate official TXT submission files.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=["auto", "yolo", "rtdetr"],
        default="auto",
        help="Ultralytics model wrapper to use. auto infers RT-DETR from an rtdetr* checkpoint name.",
    )
    parser.add_argument("--raw-root", type=Path, help="Official test root with RGB/Infrared/Depth folders.")
    parser.add_argument("--image-dir", type=Path, help="Prepared image directory. Overrides --raw-root.")
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/inference_inputs"))
    parser.add_argument("--fusion", choices=FUSION_CHOICES, default="rgb")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default=0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def zip_submission(out_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in sorted(out_dir.glob("*.txt")):
            zf.write(txt_file, arcname=txt_file.name)


def infer_model_family(weights: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if weights.name.lower().startswith("rtdetr"):
        return "rtdetr"
    return "yolo"


def create_ultralytics_model(weights: Path, family: str):
    from ultralytics import RTDETR, YOLO

    if family == "rtdetr":
        return RTDETR(str(weights))
    if family == "yolo":
        return YOLO(str(weights))
    raise ValueError(f"unsupported model family: {family}")


def main() -> None:
    args = parse_args()
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

    model_family = infer_model_family(args.weights, args.model_family)
    model = create_ultralytics_model(args.weights, model_family)
    for result in model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        augment=args.augment,
        stream=True,
        save=False,
        verbose=False,
    ):
        image_stem = Path(result.path).stem
        predictions = result_to_predictions(result)
        write_prediction_file(
            args.out_dir / f"{image_stem}.txt",
            predictions,
            max_det=args.max_det,
        )

    if args.zip_path:
        zip_submission(args.out_dir, args.zip_path)
        print(args.zip_path)
    else:
        print(args.out_dir)


if __name__ == "__main__":
    main()
