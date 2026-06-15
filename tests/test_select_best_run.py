from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_best_run_script", ROOT / "scripts" / "select_best_run.py"
)
assert SPEC and SPEC.loader
select_best_run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = select_best_run
SPEC.loader.exec_module(select_best_run)


def write_results(run_dir: Path, rows: list[tuple[int, float, float]]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    lines = ["epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n"]
    for epoch, map50, map50_95 in rows:
        lines.append(f"{epoch},{map50},{map50_95}\n")
    (run_dir / "results.csv").write_text("".join(lines), encoding="utf-8")


def test_choose_best_run_defaults_to_official_multimodal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(select_best_run, "is_run_active", lambda _name: False)
    project_dir = tmp_path / "outputs" / "runs"
    write_results(project_dir / "rgb", [(1, 0.5, 0.2), (2, 0.6, 0.4)])
    write_results(project_dir / "triad3", [(1, 0.7, 0.3)])

    best = select_best_run.choose_best_run(
        project_dir,
        [
            select_best_run.RunSpec("rgb", "rgb"),
            select_best_run.RunSpec("triad3", "triad3"),
        ],
    )

    assert best.name == "triad3"
    assert best.fusion == "triad3"
    assert best.official_multimodal is True
    assert best.best_epoch == 1
    assert best.map50_95 == 0.3


def test_choose_best_run_can_include_rgb_for_baseline_analysis(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(select_best_run, "is_run_active", lambda _name: False)
    project_dir = tmp_path / "outputs" / "runs"
    write_results(project_dir / "rgb", [(1, 0.5, 0.2), (2, 0.6, 0.4)])
    write_results(project_dir / "triad3", [(1, 0.7, 0.3)])

    best = select_best_run.choose_best_run(
        project_dir,
        [
            select_best_run.RunSpec("rgb", "rgb"),
            select_best_run.RunSpec("triad3", "triad3"),
        ],
        allow_rgb_baseline=True,
    )

    assert best.name == "rgb"
    assert best.fusion == "rgb"
    assert best.official_multimodal is False
    assert best.best_epoch == 2
    assert best.map50_95 == 0.4


def test_parse_run_spec_requires_fusion():
    spec = select_best_run.parse_run_spec("cssa:cssa3")

    assert spec.name == "cssa"
    assert spec.fusion == "cssa3"
