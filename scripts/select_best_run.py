from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from city_multimodal_detection.dataset import FUSION_CHOICES, OFFICIAL_THREE_MODAL_FUSIONS


DEFAULT_RUNS = (
    ("rgb_yolo11m_960_e100", "rgb"),
    ("triad3_yolo11m_960_e100", "triad3"),
    ("cssa3_dropout_yolo11m_960_e100", "cssa3"),
)
OFFICIAL_MULTIMODAL_FUSIONS = OFFICIAL_THREE_MODAL_FUSIONS


@dataclass(frozen=True)
class RunSpec:
    name: str
    fusion: str


@dataclass(frozen=True)
class RunSummary:
    name: str
    fusion: str
    run_dir: str
    weights: str
    active: bool
    epochs_recorded: int
    best_epoch: int
    map50: float
    map50_95: float
    official_multimodal: bool


def parse_run_spec(value: str) -> RunSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("run spec must be NAME:FUSION")
    name, fusion = value.split(":", maxsplit=1)
    if fusion not in FUSION_CHOICES:
        raise argparse.ArgumentTypeError(f"fusion must be one of {', '.join(FUSION_CHOICES)}")
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return RunSpec(name=name, fusion=fusion)


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_run_active(run_name: str) -> bool:
    try:
        completed = subprocess.run(
            ["pgrep", "-af", "scripts/train_yolo.py"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    if completed.returncode not in {0, 1}:
        return False
    return any(f"--name {run_name}" in line for line in completed.stdout.splitlines())


def summarize_run(project_dir: Path, spec: RunSpec) -> RunSummary | None:
    run_dir = project_dir / spec.name
    rows = read_rows(run_dir / "results.csv")
    weights = run_dir / "weights" / "best.pt"
    if not rows or not weights.exists():
        return None

    scored = []
    for row in rows:
        map50_95 = _to_float(row.get("metrics/mAP50-95(B)", ""))
        if map50_95 is not None:
            scored.append((map50_95, row))
    if not scored:
        return None

    _, best_row = max(scored, key=lambda item: item[0])
    best_epoch = int(float(best_row.get("epoch", "0") or 0))
    map50 = _to_float(best_row.get("metrics/mAP50(B)", "")) or 0.0
    map50_95 = _to_float(best_row.get("metrics/mAP50-95(B)", "")) or 0.0
    return RunSummary(
        name=spec.name,
        fusion=spec.fusion,
        run_dir=str(run_dir),
        weights=str(weights),
        active=is_run_active(spec.name),
        epochs_recorded=len(rows),
        best_epoch=best_epoch,
        map50=map50,
        map50_95=map50_95,
        official_multimodal=spec.fusion in OFFICIAL_MULTIMODAL_FUSIONS,
    )


def filter_official_specs(specs: list[RunSpec], allow_rgb_baseline: bool = False) -> list[RunSpec]:
    if allow_rgb_baseline:
        return specs
    return [spec for spec in specs if spec.fusion in OFFICIAL_MULTIMODAL_FUSIONS]


def choose_best_run(
    project_dir: Path,
    specs: list[RunSpec],
    allow_rgb_baseline: bool = False,
) -> RunSummary:
    candidates = filter_official_specs(specs, allow_rgb_baseline=allow_rgb_baseline)
    if not candidates:
        raise ValueError("no official multimodal candidates; pass allow_rgb_baseline=True for baseline analysis")
    summaries = [summary for spec in candidates if (summary := summarize_run(project_dir, spec))]
    if not summaries:
        raise ValueError(f"no scored runs found under {project_dir}")
    return max(summaries, key=lambda summary: summary.map50_95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose the best single training run by mAP@50-95.")
    parser.add_argument("--project-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        help="Candidate in NAME:FUSION form. Defaults to the three project runs.",
    )
    parser.add_argument(
        "--require-finished",
        action="store_true",
        help="Fail if any candidate run is still active.",
    )
    parser.add_argument(
        "--allow-rgb-baseline",
        action="store_true",
        help=(
            "Include RGB-only baseline in model selection. Do not use this for the official "
            "multimodal submission unless the organizers explicitly allow RGB-only outputs."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = args.run or [RunSpec(name=name, fusion=fusion) for name, fusion in DEFAULT_RUNS]
    candidates = filter_official_specs(specs, allow_rgb_baseline=args.allow_rgb_baseline)
    if not candidates:
        raise SystemExit("no official multimodal candidates; pass --allow-rgb-baseline for baseline analysis")
    summaries = [summary for spec in candidates if (summary := summarize_run(args.project_dir, spec))]
    if not summaries:
        raise SystemExit(f"no scored runs found under {args.project_dir}")
    active = [summary for summary in summaries if summary.active]
    if args.require_finished and active:
        names = ", ".join(summary.name for summary in active)
        raise SystemExit(f"candidate runs still active: {names}")

    best = max(summaries, key=lambda summary: summary.map50_95)
    if args.json:
        print(json.dumps(asdict(best), ensure_ascii=False, indent=2))
        return

    print(f"name={best.name}")
    print(f"fusion={best.fusion}")
    print(f"weights={best.weights}")
    print(f"best_epoch={best.best_epoch}")
    print(f"map50={best.map50:.5f}")
    print(f"map50_95={best.map50_95:.5f}")
    print(f"active={best.active}")
    print(f"official_multimodal={best.official_multimodal}")


if __name__ == "__main__":
    main()
