#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/launch_train_job.sh <run_name> -- <train_yolo.py args...>

Example:
  bash scripts/launch_train_job.sh triad3_yolo11m_960_e100 -- \
    --data datasets/city_triad3/data.yaml \
    --model weights/yolo11m.pt \
    --epochs 100 \
    --imgsz 960 \
    --batch auto-free \
    --gpu-reserve-mb 1024 \
    --max-batch-fraction 0.90 \
    --workers 16 \
    --device 0 \
    --project outputs/runs \
    --name triad3_yolo11m_960_e100 \
    --patience 30 \
    --amp \
    --no-plots
USAGE
}

if [[ $# -lt 3 || "${2:-}" != "--" ]]; then
  usage >&2
  exit 2
fi

run_name="$1"
shift 2

cd "$(dirname "$0")/.."

if pgrep -af "scripts/train_yolo.py" >/dev/null; then
  if [[ "${ALLOW_CONCURRENT_TRAIN:-0}" != "1" ]]; then
    echo "Refusing to start while another train_yolo.py job is active." >&2
    pgrep -af "scripts/train_yolo.py" >&2
    echo "Set ALLOW_CONCURRENT_TRAIN=1 only if concurrent training is intentional." >&2
    exit 1
  fi
fi

if pgrep -af "scripts/train_yolo.py" | grep -F -- "--name ${run_name}" >/dev/null; then
  echo "Refusing to start duplicate run: ${run_name}" >&2
  pgrep -af "scripts/train_yolo.py" | grep -F -- "--name ${run_name}" >&2
  exit 1
fi

mkdir -p logs pids .ultralytics
log_path="logs/${run_name}.log"
pid_path="pids/${run_name}.pid"

{
  echo "[$(date --iso-8601=seconds)] starting ${run_name}"
  echo "python scripts/train_yolo.py $*"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fi
} >> "${log_path}"

export YOLO_CONFIG_DIR="${PWD}/.ultralytics"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup .venv/bin/python scripts/train_yolo.py "$@" >> "${log_path}" 2>&1 &
pid="$!"
echo "${pid}" > "${pid_path}"
echo "started ${run_name} pid=${pid} log=${log_path}"
