from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DEPTH_ALIASES,
    IMAGE_EXTENSIONS,
    INFRARED_ALIASES,
    LABEL_ALIASES,
    RGB_ALIASES,
)
from .image_ops import (
    apply_modality_dropout,
    make_cssa_lite_image,
    make_depth_image,
    make_infrared_image,
    make_rgb_guided_depth_image,
    make_rgb_guided_ir_image,
    make_rgb_guided_rdt_image,
    make_rgb_guided_rdt_v2_image,
    make_rgb_guided_rdt_v3_image,
    make_triad_image,
    read_image,
    write_image,
)
from .yolo_io import CLASS_NAMES, make_data_yaml, read_label_file, sanitize_label_file, split_items

FUSION_CHOICES = (
    "rgb",
    "triad3",
    "cssa3",
    "ir",
    "depth",
    "rgb_guided_ir",
    "rgb_guided_depth",
    "rgb_guided_rdt",
    "rgb_guided_rdt_v2",
    "rgb_guided_rdt_v3",
)
MODALITY_DROPOUT_FUSIONS = {"triad3", "cssa3"}
OFFICIAL_THREE_MODAL_FUSIONS = {
    "triad3",
    "cssa3",
    "rgb_guided_rdt",
    "rgb_guided_rdt_v2",
    "rgb_guided_rdt_v3",
}


@dataclass(frozen=True)
class DatasetDirs:
    rgb: Path
    infrared: Path
    depth: Path
    labels: Path | None


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    rgb: Path
    infrared: Path
    depth: Path
    label: Path | None = None

    def to_json_dict(self) -> dict[str, str | None]:
        return {
            "sample_id": self.sample_id,
            "rgb": str(self.rgb),
            "infrared": str(self.infrared),
            "depth": str(self.depth),
            "label": str(self.label) if self.label else None,
        }


def _matches_alias(path: Path, aliases: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(alias.lower() == name or alias.lower() in name for alias in aliases)


def _find_dir(root: Path, aliases: tuple[str, ...]) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_dir() and _matches_alias(p, aliases)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.relative_to(root).parts), str(p).lower()))
    return candidates[0]


def infer_dataset_dirs(root: Path) -> DatasetDirs:
    root = root.resolve()
    rgb = _find_dir(root, RGB_ALIASES)
    infrared = _find_dir(root, INFRARED_ALIASES)
    depth = _find_dir(root, DEPTH_ALIASES)
    labels = _find_dir(root, LABEL_ALIASES)
    missing = [
        name
        for name, value in [("RGB", rgb), ("Infrared", infrared), ("Depth", depth)]
        if value is None
    ]
    if missing:
        raise FileNotFoundError(f"missing modality directories under {root}: {', '.join(missing)}")
    return DatasetDirs(rgb=rgb, infrared=infrared, depth=depth, labels=labels)


def _normalized_stem(path: Path) -> str:
    stem = path.stem
    lower = stem.lower()
    suffixes = (
        "_rgb",
        "-rgb",
        "_visible",
        "-visible",
        "_ir",
        "-ir",
        "_infrared",
        "-infrared",
        "_thermal",
        "-thermal",
        "_depth",
        "-depth",
    )
    for suffix in suffixes:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _index_images(root: Path) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    indexed: dict[str, Path] = {}
    for path in sorted(files):
        indexed.setdefault(_normalized_stem(path), path)
    return indexed


def _index_labels(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {p.stem: p for p in sorted(root.rglob("*.txt")) if p.is_file()}


def discover_records(root: Path, require_labels: bool = True) -> list[SampleRecord]:
    dirs = infer_dataset_dirs(root)
    rgb_files = _index_images(dirs.rgb)
    infrared_files = _index_images(dirs.infrared)
    depth_files = _index_images(dirs.depth)
    label_files = _index_labels(dirs.labels)

    common_ids = sorted(set(rgb_files) & set(infrared_files) & set(depth_files))
    records: list[SampleRecord] = []
    for sample_id in common_ids:
        label = label_files.get(sample_id)
        if require_labels and label is None:
            continue
        if label is not None:
            read_label_file(label, clip_boxes=True)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                rgb=rgb_files[sample_id],
                infrared=infrared_files[sample_id],
                depth=depth_files[sample_id],
                label=label,
            )
        )
    return records


def copy_or_link_file(source: Path, target: Path, strategy: str = "copy") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if strategy == "copy":
        shutil.copy2(source, target)
    elif strategy == "hardlink":
        os.link(source, target)
    elif strategy == "symlink":
        target.symlink_to(source)
    else:
        raise ValueError(f"unsupported link strategy: {strategy}")


def materialize_record_image(
    record: SampleRecord,
    output_image: Path,
    fusion: str = "rgb",
    link_strategy: str = "copy",
    use_clahe: bool = True,
    drop_channels: tuple[int, ...] | None = None,
) -> None:
    if fusion == "rgb":
        if drop_channels:
            raise ValueError("modality dropout is only supported for fused 3-channel images")
        copy_or_link_file(record.rgb, output_image, strategy=link_strategy)
        return
    if fusion not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {', '.join(FUSION_CHOICES)}")
    if drop_channels and fusion not in MODALITY_DROPOUT_FUSIONS:
        raise ValueError("modality dropout is only supported for triad3/cssa3 modality-channel images")

    rgb = read_image(record.rgb)
    infrared = read_image(record.infrared)
    depth = read_image(record.depth, unchanged=True)
    if fusion == "triad3":
        image = make_triad_image(rgb, infrared, depth, use_clahe=use_clahe)
    elif fusion == "cssa3":
        image = make_cssa_lite_image(rgb, infrared, depth, use_clahe=use_clahe)
    elif fusion == "ir":
        image = make_infrared_image(infrared, use_clahe=use_clahe)
    elif fusion == "depth":
        image = make_depth_image(depth)
    elif fusion == "rgb_guided_ir":
        image = make_rgb_guided_ir_image(rgb, infrared, use_clahe=use_clahe)
    elif fusion == "rgb_guided_depth":
        image = make_rgb_guided_depth_image(rgb, depth)
    elif fusion == "rgb_guided_rdt":
        image = make_rgb_guided_rdt_image(rgb, infrared, depth, use_clahe=use_clahe)
    elif fusion == "rgb_guided_rdt_v2":
        image = make_rgb_guided_rdt_v2_image(rgb, infrared, depth, use_clahe=use_clahe)
    elif fusion == "rgb_guided_rdt_v3":
        image = make_rgb_guided_rdt_v3_image(rgb, infrared, depth, use_clahe=use_clahe)
    else:  # pragma: no cover - guarded by FUSION_CHOICES.
        raise ValueError(f"unsupported fusion: {fusion}")
    if drop_channels:
        image = apply_modality_dropout(image, drop_channels)
    write_image(output_image, image)


def _prepare_record_job(
    output_root: Path,
    split: str,
    record: SampleRecord,
    fusion: str,
    link_strategy: str,
    require_labels: bool,
    variant_suffix: str = "",
    drop_channels: tuple[int, ...] | None = None,
) -> dict[str, str | None]:
    image_suffix = record.rgb.suffix if fusion == "rgb" else ".png"
    output_id = f"{record.sample_id}{variant_suffix}"
    target_image = output_root / "images" / split / f"{output_id}{image_suffix}"
    materialize_record_image(
        record,
        target_image,
        fusion=fusion,
        link_strategy=link_strategy,
        drop_channels=drop_channels,
    )
    if require_labels and record.label is not None:
        target_label = output_root / "labels" / split / f"{output_id}.txt"
        sanitize_label_file(record.label, target_label)
    payload = record.to_json_dict()
    payload["output_id"] = output_id
    payload["split"] = split
    payload["fusion"] = fusion
    payload["variant"] = variant_suffix.lstrip("_") or "base"
    return payload


def prepare_yolo_dataset(
    raw_root: Path,
    output_root: Path,
    fusion: str = "rgb",
    val_ratio: float = 0.2,
    seed: int = 42,
    link_strategy: str = "copy",
    require_labels: bool = True,
    workers: int = 1,
    modality_dropout: bool = False,
) -> Path:
    records = discover_records(raw_root, require_labels=require_labels)
    if not records:
        raise ValueError(f"no usable records discovered under {raw_root}")

    if require_labels:
        train_records, val_records = split_items(records, val_ratio=val_ratio, seed=seed)
        splits = {"train": train_records}
        if val_records:
            splits["val"] = val_records
    else:
        splits = {"test": records}

    output_root.mkdir(parents=True, exist_ok=True)
    if fusion not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {', '.join(FUSION_CHOICES)}")
    if modality_dropout and fusion not in MODALITY_DROPOUT_FUSIONS:
        raise ValueError("modality_dropout is only valid for triad3/cssa3 modality-channel fusions")

    jobs = []
    dropout_variants = [
        ("_drop_rgb", (0,)),
        ("_drop_ir", (1,)),
        ("_drop_depth", (2,)),
    ]
    for split, split_records in splits.items():
        for record in split_records:
            jobs.append((output_root, split, record, fusion, link_strategy, require_labels, "", None))
            if modality_dropout and split == "train" and fusion != "rgb":
                for suffix, channels in dropout_variants:
                    jobs.append(
                        (output_root, split, record, fusion, link_strategy, require_labels, suffix, channels)
                    )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            payloads = list(executor.map(lambda args: _prepare_record_job(*args), jobs))
    else:
        payloads = [_prepare_record_job(*job) for job in jobs]

    manifest_path = output_root / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for payload in payloads:
            manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if require_labels:
        data_yaml = output_root / "data.yaml"
        val_path = "images/val" if "val" in splits else "images/train"
        data_yaml.write_text(
            make_data_yaml(output_root, CLASS_NAMES, val_path=val_path, channels=3),
            encoding="utf-8",
        )
        return data_yaml
    return manifest_path
