from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an Ultralytics training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", help="Optional name used to search active train_yolo processes.")
    parser.add_argument("--tail", type=int, default=5)
    return parser.parse_args()


def read_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def get_active_processes(run_name: str | None) -> list[str]:
    if not run_name:
        return []
    try:
        completed = subprocess.run(
            ["pgrep", "-af", "scripts/train_yolo.py"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return []
    if completed.returncode not in {0, 1}:
        return []
    return [line for line in completed.stdout.splitlines() if f"--name {run_name}" in line]


def main() -> None:
    args = parse_args()
    rows = read_rows(args.run_dir / "results.csv")
    processes = get_active_processes(args.run_name)
    weights_dir = args.run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    print(f"run_dir: {args.run_dir}")
    print(f"active: {bool(processes)}")
    if processes:
        print(f"processes: {len(processes)} matching train_yolo entries")
    print(f"epochs_recorded: {len(rows)}")
    if rows:
        latest = rows[-1]
        print(
            "latest: "
            f"epoch={latest.get('epoch', '')} "
            f"mAP50={latest.get('metrics/mAP50(B)', '')} "
            f"mAP50-95={latest.get('metrics/mAP50-95(B)', '')} "
            f"precision={latest.get('metrics/precision(B)', '')} "
            f"recall={latest.get('metrics/recall(B)', '')}"
        )
        scored = [row for row in rows if row.get("metrics/mAP50-95(B)", "")]
        if scored:
            best_row = max(scored, key=lambda row: float(row["metrics/mAP50-95(B)"]))
            print(
                "best_row: "
                f"epoch={best_row.get('epoch', '')} "
                f"mAP50={best_row.get('metrics/mAP50(B)', '')} "
                f"mAP50-95={best_row.get('metrics/mAP50-95(B)', '')}"
            )
    print(f"best_pt: {best.exists()} {best}")
    print(f"last_pt: {last.exists()} {last}")

    if args.tail and rows:
        print("tail:")
        fieldnames = [
            "epoch",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(B)",
            "metrics/recall(B)",
        ]
        for row in rows[-args.tail :]:
            print(", ".join(f"{name}={row.get(name, '')}" for name in fieldnames))


if __name__ == "__main__":
    main()
