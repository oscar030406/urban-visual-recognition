from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class AP and YOLO label size distribution.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--project", default="outputs/analysis")
    parser.add_argument("--name", default=None)
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
    return (data_yaml.parent / raw_path).resolve() if not raw_path.exists() else raw_path.resolve()


def names_from_cfg(cfg: dict[str, Any]) -> dict[int, str]:
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def label_dir_for_split(root: Path, cfg: dict[str, Any], split: str) -> Path:
    image_rel = Path(str(cfg[split]))
    if image_rel.is_absolute():
        labels = Path(str(image_rel).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
    else:
        labels = root / Path(str(image_rel).replace("images", "labels", 1))
    return labels


def classify_size(width: float, height: float, imgsz: int) -> str:
    area = width * height * imgsz * imgsz
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def collect_label_stats(label_dir: Path, nc: int, imgsz: int) -> dict[int, Counter]:
    stats: dict[int, Counter] = {class_id: Counter() for class_id in range(nc)}
    for label_path in sorted(label_dir.glob("*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            width = float(parts[3])
            height = float(parts[4])
            if 0 <= class_id < nc:
                stats[class_id]["instances"] += 1
                stats[class_id][classify_size(width, height, imgsz)] += 1
    return stats


def to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (float, int)):
        return [float(value)]
    return [float(item) for item in value]


def metric_rows(metrics: Any, names: dict[int, str], label_stats: dict[int, Counter]) -> list[dict[str, Any]]:
    box = metrics.box
    nc = len(names)
    rows_by_class: dict[int, dict[str, Any]] = {
        class_id: {
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "instances": label_stats[class_id]["instances"],
            "small": label_stats[class_id]["small"],
            "medium": label_stats[class_id]["medium"],
            "large": label_stats[class_id]["large"],
            "small_ratio": "",
            "medium_ratio": "",
            "large_ratio": "",
            "precision": "",
            "recall": "",
            "ap50": "",
            "ap50_95": "",
        }
        for class_id in range(nc)
    }

    for class_id, counter in label_stats.items():
        total = counter["instances"]
        if total:
            rows_by_class[class_id]["small_ratio"] = counter["small"] / total
            rows_by_class[class_id]["medium_ratio"] = counter["medium"] / total
            rows_by_class[class_id]["large_ratio"] = counter["large"] / total

    ap_class_index = [int(item) for item in to_list(getattr(box, "ap_class_index", []))]
    precision = to_list(getattr(box, "p", []))
    recall = to_list(getattr(box, "r", []))
    all_ap = getattr(box, "all_ap", None)
    if hasattr(all_ap, "tolist"):
        all_ap = all_ap.tolist()
    maps = to_list(getattr(box, "maps", []))

    for offset, class_id in enumerate(ap_class_index):
        if class_id not in rows_by_class:
            continue
        if offset < len(precision):
            rows_by_class[class_id]["precision"] = precision[offset]
        if offset < len(recall):
            rows_by_class[class_id]["recall"] = recall[offset]
        if all_ap and offset < len(all_ap):
            class_ap = [float(item) for item in all_ap[offset]]
            if class_ap:
                rows_by_class[class_id]["ap50"] = class_ap[0]
                rows_by_class[class_id]["ap50_95"] = sum(class_ap) / len(class_ap)
        elif class_id < len(maps):
            rows_by_class[class_id]["ap50_95"] = maps[class_id]

    return [rows_by_class[class_id] for class_id in range(nc)]


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs(
    rows: list[dict[str, Any]],
    out_csv: Path,
    out_md: Path,
    weights: Path,
    data: Path,
    split: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(
        rows,
        key=lambda row: (float(row["ap50_95"]) if row["ap50_95"] != "" else -1.0),
    )
    with out_md.open("w", encoding="utf-8") as fp:
        fp.write("# Class / Size AP Analysis\n\n")
        fp.write(f"- weights: `{weights}`\n")
        fp.write(f"- data: `{data}`\n")
        fp.write(f"- split: `{split}`\n\n")
        fp.write("## Lowest AP Classes\n\n")
        fp.write("| class | instances | small% | medium% | large% | AP50 | AP50-95 |\n")
        fp.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted_rows:
            total = row["instances"] or 0
            fp.write(
                f"| {row['class_id']} {row['class_name']} | {total} | "
                f"{fmt(row['small_ratio'])} | {fmt(row['medium_ratio'])} | {fmt(row['large_ratio'])} | "
                f"{fmt(row['ap50'])} | {fmt(row['ap50_95'])} |\n"
            )
        fp.write("\n## JSON Rows\n\n")
        fp.write("```json\n")
        fp.write(json.dumps(rows, ensure_ascii=False, indent=2))
        fp.write("\n```\n")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = read_yaml(args.data)
    names = names_from_cfg(cfg)
    root = dataset_root(args.data, cfg)
    label_dir = label_dir_for_split(root, cfg, args.split)
    label_stats = collect_label_stats(label_dir, nc=len(names), imgsz=args.imgsz)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=args.name,
        plots=False,
    )
    rows = metric_rows(metrics, names, label_stats)
    write_outputs(rows, args.out_csv, args.out_md, args.weights, args.data, args.split)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    print(f"mAP@50: {metrics.box.map50:.6f}")
    print(f"mAP@50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
