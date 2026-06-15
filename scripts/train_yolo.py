from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

BatchValue = int | float | Literal["auto-free"]


def parse_batch(value: str) -> BatchValue:
    normalized = value.strip().lower()
    if normalized in {"auto-free", "auto_remaining", "auto-remaining", "free"}:
        return "auto-free"
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _single_device_id(device: str | int) -> str | None:
    value = str(device).strip()
    if value.lower() in {"cpu", "mps"} or "," in value:
        return None
    return value


def query_gpu_memory_mb(device: str | int) -> tuple[int, int] | None:
    device_id = _single_device_id(device)
    if device_id is None:
        return None
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_id}",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first_line = completed.stdout.splitlines()[0]
    total_text, used_text = [part.strip() for part in first_line.split(",", maxsplit=1)]
    return int(total_text), int(used_text)


def auto_free_batch_fraction(
    total_mb: int,
    used_mb: int,
    reserve_mb: int = 1024,
    max_fraction: float = 0.90,
) -> float:
    if total_mb <= 0:
        raise ValueError("total GPU memory must be positive")
    usable_mb = total_mb - used_mb - reserve_mb
    if usable_mb <= 0:
        raise RuntimeError(
            f"not enough free GPU memory: total={total_mb}MiB used={used_mb}MiB reserve={reserve_mb}MiB"
        )
    fraction = usable_mb / total_mb
    return round(max(0.05, min(max_fraction, fraction)), 3)


def resolve_batch(
    batch: BatchValue,
    device: str | int,
    reserve_mb: int,
    max_fraction: float,
) -> int | float:
    if batch != "auto-free":
        return batch
    memory = query_gpu_memory_mb(device)
    if memory is None:
        print("auto-free batch requested but GPU memory could not be queried; falling back to -1")
        return -1
    total_mb, used_mb = memory
    fraction = auto_free_batch_fraction(
        total_mb=total_mb,
        used_mb=used_mb,
        reserve_mb=reserve_mb,
        max_fraction=max_fraction,
    )
    print(
        "auto-free batch resolved: "
        f"total={total_mb}MiB used={used_mb}MiB reserve={reserve_mb}MiB "
        f"max_fraction={max_fraction:.2f} batch={fraction}"
    )
    return fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics detector.")
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml.")
    parser.add_argument(
        "--model",
        default="weights/yolo11m.pt",
        help="Local Ultralytics checkpoint path. Keep this local to avoid online downloads during training.",
    )
    parser.add_argument(
        "--model-family",
        choices=["auto", "yolo", "rtdetr"],
        default="auto",
        help="Ultralytics model wrapper to use. auto infers RT-DETR from an rtdetr* checkpoint name.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional local checkpoint to load after constructing a YAML model.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default="auto-free",
        help=(
            "Integer batch size, -1 for 60%% Ultralytics auto-batch, a GPU memory fraction "
            "such as 0.85, or auto-free to use currently free GPU memory minus reserve."
        ),
    )
    parser.add_argument("--device", default=0)
    parser.add_argument("--gpu-reserve-mb", type=int, default=1024)
    parser.add_argument("--max-batch-fraction", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--project", default="outputs/runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR fraction override.")
    parser.add_argument("--warmup-epochs", dest="warmup_epochs", type=float, default=None)
    parser.add_argument("--warmup-momentum", dest="warmup_momentum", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", dest="warmup_bias_lr", type=float, default=None)
    parser.add_argument("--box", type=float, default=None, help="Box loss weight override.")
    parser.add_argument("--cls", type=float, default=None, help="Class loss weight override.")
    parser.add_argument("--dfl", type=float, default=None, help="DFL loss weight override.")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument(
        "--gate-stem",
        choices=["none", "rgb_ir_depth5"],
        default="none",
        help="Replace YOLO layer 0 with an RGB-main IR/depth feature gate stem.",
    )
    parser.add_argument("--gate-alpha-init", type=float, default=0.1)
    parser.add_argument(
        "--aux-gate",
        choices=["none", "p3p4p5"],
        default="none",
        help="Use 5-channel input but keep YOLO RGB-main and gate P3/P4/P5 with IR/depth.",
    )
    parser.add_argument("--aux-gate-alpha-init", type=float, default=0.1)
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability.")
    parser.add_argument("--copy-paste", dest="copy_paste", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None, help="Image scale augmentation factor.")
    parser.add_argument("--close-mosaic", dest="close_mosaic", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None, help="Number of early layers to freeze.")
    parser.add_argument(
        "--optimizer",
        default=None,
        help="Ultralytics optimizer. Use AdamW/SGD/etc. to prevent optimizer=auto from overriding lr0.",
    )
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_local_model(model: str) -> str:
    path = Path(model)
    if path.suffix == ".pt" and not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. "
            "Download public pretrained weights during environment setup, then train with a local path."
        )
    return str(path)


def ensure_local_checkpoint(path_text: str | None) -> str | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.suffix == ".pt" and not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}. "
            "Download public pretrained weights during environment setup, then train with a local path."
        )
    return str(path)


def infer_model_family(model: str, requested: str) -> str:
    if requested != "auto":
        return requested
    name = Path(model).name.lower()
    if name.startswith("rtdetr"):
        return "rtdetr"
    return "yolo"


def create_ultralytics_model(model: str, family: str):
    from ultralytics import RTDETR, YOLO

    if family == "rtdetr":
        return RTDETR(model)
    if family == "yolo":
        return YOLO(model)
    raise ValueError(f"unsupported model family: {family}")


def remap_shifted_final_layer_weights(model, checkpoint: str) -> int:
    """Load shifted head weights when YAML insertions move pretrained layers."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_model = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else None
    if source_model is None or not hasattr(source_model, "state_dict"):
        return 0
    source_state = source_model.float().state_dict()
    target_state = model.model.state_dict()
    target_detect_index = len(model.model.model) - 1
    target_prefix = f"model.{target_detect_index}."
    source_prefix = "model.23."
    if target_prefix == source_prefix:
        return 0

    source_detect = source_model.model[-1]
    target_detect = model.model.model[-1]
    source_branches = len(getattr(source_detect, "cv2", []))
    target_branches = len(getattr(target_detect, "cv2", []))
    branch_offset = max(0, target_branches - source_branches)

    mapped = {}
    if target_branches == source_branches + 1:
        p2_shifted_layers = {
            17: 23,  # old P3->P4 downsample conv
            19: 25,  # old P4 C3k2
            20: 26,  # old P4->P5 downsample conv
            22: 28,  # old P5 C3k2
        }
        for source_index, target_index in p2_shifted_layers.items():
            source_layer_prefix = f"model.{source_index}."
            target_layer_prefix = f"model.{target_index}."
            for source_key, source_value in source_state.items():
                if not source_key.startswith(source_layer_prefix):
                    continue
                target_key = target_layer_prefix + source_key.removeprefix(source_layer_prefix)
                target_value = target_state.get(target_key)
                if target_value is not None and tuple(target_value.shape) == tuple(source_value.shape):
                    mapped[target_key] = source_value.to(dtype=target_value.dtype)

    for source_key, source_value in source_state.items():
        if not source_key.startswith(source_prefix):
            continue
        suffix = source_key.removeprefix(source_prefix)
        branch_match = re.match(r"(cv[23])\.(\d+)\.(.+)", suffix)
        if branch_match:
            module_name, branch_text, rest = branch_match.groups()
            target_branch = int(branch_text) + branch_offset
            suffix = f"{module_name}.{target_branch}.{rest}"
        target_key = target_prefix + suffix
        target_value = target_state.get(target_key)
        if target_value is not None and tuple(target_value.shape) == tuple(source_value.shape):
            mapped[target_key] = source_value.to(dtype=target_value.dtype)
    if not mapped:
        return 0
    target_state.update(mapped)
    model.model.load_state_dict(target_state, strict=False)
    return len(mapped)


def append_experiment_log(args: argparse.Namespace, metrics) -> None:
    log_path = Path("docs/experiment_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "time",
                "run",
                "model",
                "pretrained",
                "data",
                "epochs",
                "imgsz",
                "batch",
                "map50",
                "map50_95",
                "notes",
            ],
        )
        if not exists:
            writer.writeheader()
        box = getattr(metrics, "box", None)
        writer.writerow(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "run": args.name or "",
                "model": args.model,
                "pretrained": args.pretrained or "",
                "data": str(args.data),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "map50": getattr(box, "map50", ""),
                "map50_95": getattr(box, "map", ""),
                "notes": (
                    f"lr0={args.lr0},lrf={args.lrf},mosaic={args.mosaic},"
                    f"mixup={args.mixup},copy_paste={args.copy_paste},scale={args.scale},"
                    f"close_mosaic={args.close_mosaic},freeze={args.freeze},"
                    f"optimizer={args.optimizer},cos_lr={args.cos_lr},"
                    f"warmup_epochs={args.warmup_epochs},warmup_momentum={args.warmup_momentum},"
                    f"warmup_bias_lr={args.warmup_bias_lr},box={args.box},cls={args.cls},"
                    f"dfl={args.dfl},weight_decay={args.weight_decay},"
                    f"gate_stem={args.gate_stem},gate_alpha_init={args.gate_alpha_init},"
                    f"aux_gate={args.aux_gate},aux_gate_alpha_init={args.aux_gate_alpha_init}"
                ),
            }
        )


def optional_train_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    optional_fields = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "weight_decay": args.weight_decay,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "scale": args.scale,
        "close_mosaic": args.close_mosaic,
        "freeze": args.freeze,
        "optimizer": args.optimizer,
        "cos_lr": args.cos_lr,
    }
    train_kwargs = {key: value for key, value in optional_fields.items() if value is not None}
    return train_kwargs


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from city_multimodal_detection.yolo_attention import (
        attach_aux_pyramid_gates,
        copy_gate_stem_rgb_weights_from_model,
        register_ultralytics_attention,
        replace_first_conv_with_gate_stem,
    )

    register_ultralytics_attention()
    args.batch = resolve_batch(
        args.batch,
        device=args.device,
        reserve_mb=args.gpu_reserve_mb,
        max_fraction=args.max_batch_fraction,
    )
    model_path = resolve_local_model(args.model)
    model_family = infer_model_family(model_path, args.model_family)
    if args.gate_stem != "none" and model_family != "yolo":
        raise SystemExit("--gate-stem is only supported for YOLO models")
    if args.aux_gate != "none" and model_family != "yolo":
        raise SystemExit("--aux-gate is only supported for YOLO models")
    if args.aux_gate != "none" and args.gate_stem != "none":
        raise SystemExit("--aux-gate and --gate-stem are mutually exclusive")
    model = create_ultralytics_model(model_path, model_family)
    pretrained = ensure_local_checkpoint(args.pretrained)
    trainer_class = None
    if args.gate_stem == "rgb_ir_depth5":
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel

        gate_alpha_init = args.gate_alpha_init

        class RGBIRDepthGateTrainer(DetectionTrainer):
            def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
                gate_model = DetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                )
                replace_first_conv_with_gate_stem(gate_model, alpha=gate_alpha_init)
                if weights:
                    gate_model.load(weights)
                    copied = copy_gate_stem_rgb_weights_from_model(gate_model, weights)
                    print(f"loaded gate-stem RGB weights: {copied} tensors")
                return gate_model

        trainer_class = RGBIRDepthGateTrainer
        if pretrained is not None:
            train_kwargs = optional_train_kwargs(args)
            train_kwargs["pretrained"] = pretrained
        else:
            train_kwargs = optional_train_kwargs(args)
        print(f"enabled RGB/IR/depth gate stem with alpha_init={args.gate_alpha_init}")
    elif args.aux_gate == "p3p4p5":
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel

        aux_gate_alpha_init = args.aux_gate_alpha_init

        class AuxPyramidGateTrainer(DetectionTrainer):
            def get_model(self, cfg: str | None = None, weights=None, verbose: bool = True):
                # Keep the pretrained detector strictly RGB-main. The first
                # layer pre-hook slices 5-channel input to RGB while keeping
                # IR/depth available for P3/P4/P5 gates.
                aux_model = DetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=3,
                    verbose=verbose,
                )
                gates = attach_aux_pyramid_gates(aux_model, alpha=aux_gate_alpha_init)
                if weights:
                    aux_model.load(weights)
                    print(f"loaded RGB-main weights with AuxPyramidGates on layers {sorted(gates.layer_channels)}")
                return aux_model

        trainer_class = AuxPyramidGateTrainer
        if pretrained is not None:
            train_kwargs = optional_train_kwargs(args)
            train_kwargs["pretrained"] = pretrained
        else:
            train_kwargs = optional_train_kwargs(args)
        print(
            "enabled AuxPyramidGates on RGB-main model layers "
            f"[16, 19, 22] with alpha_init={args.aux_gate_alpha_init}"
        )
    elif pretrained is not None:
        model.load(pretrained)
        remapped = remap_shifted_final_layer_weights(model, pretrained)
        if remapped:
            print(f"remapped shifted final-layer weights: {remapped} tensors")
        train_kwargs = optional_train_kwargs(args)
    else:
        train_kwargs = optional_train_kwargs(args)
    if train_kwargs:
        print(f"training overrides: {train_kwargs}")
    train_results = model.train(
        trainer=trainer_class,
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        amp=args.amp,
        plots=args.plots,
        task="detect",
        **train_kwargs,
    )
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    append_experiment_log(args, metrics)
    print(train_results)
    print(f"mAP@50: {metrics.box.map50:.6f}")
    print(f"mAP@50-95: {metrics.box.map:.6f}")


if __name__ == "__main__":
    main()
