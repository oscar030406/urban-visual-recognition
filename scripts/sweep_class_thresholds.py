from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.ultralytics_adapter import result_to_predictions


IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy search per-class confidence thresholds on a YOLO val split.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1408)
    parser.add_argument("--base-conf", type=float, default=0.0005)
    parser.add_argument("--init-conf", type=float, default=0.00125)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--candidate-max-det", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=0)
    parser.add_argument("--thresholds", type=parse_float_list, default=[0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.003])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--box-penalty", type=float, default=0.0, help="Subtract penalty * boxes_per_image from objective.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def dataset_root(data_yaml: Path, cfg: dict[str, Any]) -> Path:
    raw_path = Path(str(cfg.get("path", data_yaml.parent)))
    if raw_path.is_absolute():
        return raw_path
    candidate = (Path.cwd() / raw_path).resolve()
    if candidate.exists():
        return candidate
    return (data_yaml.parent / raw_path).resolve()


def names_from_cfg(cfg: dict[str, Any]) -> dict[int, str]:
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def split_path(root: Path, cfg: dict[str, Any], split: str) -> Path:
    value = Path(str(cfg[split]))
    return value if value.is_absolute() else root / value


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def xywhn_to_xyxy(values: list[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = values[:4]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(label_dir: Path, num_classes: int) -> tuple[dict[int, dict[str, list[tuple[float, float, float, float]]]], dict[int, int]]:
    gt: dict[int, dict[str, list[tuple[float, float, float, float]]]] = {
        class_id: defaultdict(list) for class_id in range(num_classes)
    }
    counts = {class_id: 0 for class_id in range(num_classes)}
    for label_path in sorted(label_dir.glob("*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if not 0 <= class_id < num_classes:
                continue
            box = xywhn_to_xyxy([float(item) for item in parts[1:5]])
            gt[class_id][label_path.stem].append(box)
            counts[class_id] += 1
    return gt, counts


def load_or_predict(args: argparse.Namespace, image_dir: Path) -> list[dict[str, Any]]:
    if args.cache.exists():
        with args.cache.open("rb") as fp:
            return pickle.load(fp)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    predictions: list[dict[str, Any]] = []
    for result in model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.base_conf,
        iou=args.iou,
        max_det=args.candidate_max_det,
        device=args.device,
        augment=args.augment,
        stream=True,
        save=False,
        verbose=False,
    ):
        stem = Path(result.path).stem
        for pred in result_to_predictions(result):
            predictions.append(
                {
                    "image": stem,
                    "class_id": int(pred.class_id),
                    "box": xywhn_to_xyxy([pred.cx, pred.cy, pred.w, pred.h]),
                    "confidence": float(pred.confidence),
                }
            )
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as fp:
        pickle.dump(predictions, fp)
    return predictions


def apply_thresholds(
    predictions: list[dict[str, Any]],
    thresholds: list[float],
    max_det: int,
) -> list[dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        class_id = int(pred["class_id"])
        if 0 <= class_id < len(thresholds) and pred["confidence"] >= thresholds[class_id]:
            by_image[str(pred["image"])].append(pred)
    filtered: list[dict[str, Any]] = []
    for image_preds in by_image.values():
        image_preds.sort(key=lambda item: float(item["confidence"]), reverse=True)
        filtered.extend(image_preds[:max_det])
    return filtered


def ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    return float(np.mean([np.max(precision[recall >= t]) if np.any(recall >= t) else 0.0 for t in np.linspace(0, 1, 101)]))


def evaluate_ap(
    predictions: list[dict[str, Any]],
    gt: dict[int, dict[str, list[tuple[float, float, float, float]]]],
    gt_counts: dict[int, int],
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    preds_by_class: dict[int, list[dict[str, Any]]] = {class_id: [] for class_id in range(num_classes)}
    for pred in predictions:
        class_id = int(pred["class_id"])
        if 0 <= class_id < num_classes:
            preds_by_class[class_id].append(pred)

    class_maps: dict[int, float] = {}
    class_iou_aps: list[float] = []
    for class_id in range(num_classes):
        total_gt = gt_counts.get(class_id, 0)
        if total_gt == 0:
            continue
        class_preds = sorted(preds_by_class[class_id], key=lambda item: float(item["confidence"]), reverse=True)
        iou_aps = []
        for iou_threshold in IOU_THRESHOLDS:
            matched = {image: set() for image in gt[class_id]}
            tp = np.zeros(len(class_preds), dtype=np.float32)
            fp = np.zeros(len(class_preds), dtype=np.float32)
            for index, pred in enumerate(class_preds):
                image = str(pred["image"])
                candidates = gt[class_id].get(image, [])
                best_iou = 0.0
                best_gt = -1
                for gt_index, gt_box in enumerate(candidates):
                    if gt_index in matched.get(image, set()):
                        continue
                    current_iou = box_iou(pred["box"], gt_box)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt = gt_index
                if best_iou >= iou_threshold and best_gt >= 0:
                    tp[index] = 1
                    matched[image].add(best_gt)
                else:
                    fp[index] = 1
            if len(class_preds) == 0:
                iou_aps.append(0.0)
                continue
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / max(total_gt, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            iou_aps.append(ap_101(recall, precision))
        class_map = float(np.mean(iou_aps))
        class_maps[class_id] = class_map
        class_iou_aps.append(class_map)
    return float(np.mean(class_iou_aps)) if class_iou_aps else 0.0, class_maps


def summarize_counts(predictions: list[dict[str, Any]], num_classes: int) -> dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for pred in predictions:
        class_id = int(pred["class_id"])
        if 0 <= class_id < num_classes:
            counts[class_id] += 1
    return counts


def objective(score: float, predictions: list[dict[str, Any]], image_count: int, box_penalty: float) -> float:
    return score - box_penalty * (len(predictions) / max(image_count, 1))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))

    cfg = read_yaml(args.data)
    names = names_from_cfg(cfg)
    num_classes = len(names)
    root = dataset_root(args.data, cfg)
    image_dir = split_path(root, cfg, "val")
    label_dir = label_dir_from_image_dir(image_dir)
    image_count = len(list(image_dir.glob("*")))
    gt, gt_counts = load_ground_truth(label_dir, num_classes)
    raw_predictions = load_or_predict(args, image_dir)

    thresholds = [float(args.init_conf)] * num_classes
    rows: list[dict[str, Any]] = []

    def record(stage: str, test_class: int | None, test_threshold: float | None, candidate_thresholds: list[float]) -> tuple[float, float]:
        filtered = apply_thresholds(raw_predictions, candidate_thresholds, max_det=args.max_det)
        score, class_maps = evaluate_ap(filtered, gt, gt_counts, num_classes)
        obj = objective(score, filtered, image_count, args.box_penalty)
        counts = summarize_counts(filtered, num_classes)
        rows.append(
            {
                "stage": stage,
                "class_id": "" if test_class is None else test_class,
                "class_name": "" if test_class is None else names.get(test_class, str(test_class)),
                "threshold": "" if test_threshold is None else test_threshold,
                "map50_95": score,
                "objective": obj,
                "total_boxes": len(filtered),
                "boxes_per_image": len(filtered) / max(image_count, 1),
                "thresholds": json.dumps(candidate_thresholds),
                "class_maps": json.dumps(class_maps, sort_keys=True),
                "class_box_counts": json.dumps(counts, sort_keys=True),
            }
        )
        return score, obj

    best_score, best_obj = record("initial", None, None, thresholds)
    for round_index in range(args.rounds):
        improved = False
        for class_id in range(num_classes):
            best_for_class = thresholds[class_id]
            best_for_class_score = best_score
            best_for_class_obj = best_obj
            for candidate in args.thresholds:
                candidate_thresholds = list(thresholds)
                candidate_thresholds[class_id] = float(candidate)
                score, obj = record(f"round_{round_index + 1}", class_id, float(candidate), candidate_thresholds)
                if obj > best_for_class_obj + 1e-9:
                    best_for_class = float(candidate)
                    best_for_class_score = score
                    best_for_class_obj = obj
            if best_for_class != thresholds[class_id]:
                thresholds[class_id] = best_for_class
                best_score = best_for_class_score
                best_obj = best_for_class_obj
                improved = True
                record(f"accepted_round_{round_index + 1}", class_id, best_for_class, thresholds)
        if not improved:
            break

    final_predictions = apply_thresholds(raw_predictions, thresholds, max_det=args.max_det)
    final_score, final_class_maps = evaluate_ap(final_predictions, gt, gt_counts, num_classes)
    final_counts = summarize_counts(final_predictions, num_classes)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "weights": str(args.weights),
                "data": str(args.data),
                "imgsz": args.imgsz,
                "base_conf": args.base_conf,
                "init_conf": args.init_conf,
                "iou": args.iou,
                "augment": args.augment,
                "max_det": args.max_det,
                "candidate_max_det": args.candidate_max_det,
                "box_penalty": args.box_penalty,
                "class_conf": thresholds,
                "map50_95": final_score,
                "objective": objective(final_score, final_predictions, image_count, args.box_penalty),
                "total_boxes": len(final_predictions),
                "boxes_per_image": len(final_predictions) / max(image_count, 1),
                "class_maps": final_class_maps,
                "class_box_counts": final_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w", encoding="utf-8") as fp:
        fp.write("# Per-Class Threshold Search\n\n")
        fp.write(f"- weights: `{args.weights}`\n")
        fp.write(f"- data: `{args.data}`\n")
        fp.write(f"- imgsz/conf/iou/TTA: `{args.imgsz}` / `{args.base_conf}` / `{args.iou}` / `{args.augment}`\n")
        fp.write(f"- final mAP@50-95: `{final_score:.6f}`\n")
        fp.write(f"- total boxes on val: `{len(final_predictions)}`\n\n")
        fp.write("| class | threshold | AP50-95 | val boxes | gt instances |\n")
        fp.write("|---|---:|---:|---:|---:|\n")
        for class_id in range(num_classes):
            fp.write(
                f"| {class_id} {names.get(class_id, str(class_id))} | "
                f"{thresholds[class_id]:.6g} | "
                f"{final_class_maps.get(class_id, 0.0):.6f} | "
                f"{final_counts.get(class_id, 0)} | "
                f"{gt_counts.get(class_id, 0)} |\n"
            )

    print(f"final mAP@50-95: {final_score:.6f}")
    print(f"thresholds: {thresholds}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

