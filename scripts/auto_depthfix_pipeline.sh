#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-cssa3_depthfix_dropout_yolo11m_960_e140}"
RAW_TRAIN="${RAW_TRAIN:-data/训练集/AIC2026_Train_2000}"
RAW_TEST="${RAW_TEST:-data/测试集/AIC2026_PHASE_1_1000}"
DATASET_DIR="${DATASET_DIR:-datasets/city_cssa3_depthfix_dropout}"
FUSION="${FUSION:-cssa3}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-1}"
MODEL="${MODEL:-weights/yolo11m.pt}"
EPOCHS="${EPOCHS:-140}"
IMGSZ="${IMGSZ:-960}"
WORKERS="${WORKERS:-16}"
DEVICE="${DEVICE:-0}"
PATIENCE="${PATIENCE:-40}"
GPU_RESERVE_MB="${GPU_RESERVE_MB:-1536}"
MAX_BATCH_FRACTION="${MAX_BATCH_FRACTION:-0.88}"
CONF="${CONF:-0.001}"
IOU="${IOU:-0.7}"
MAX_DET="${MAX_DET:-100}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

LOG_DIR="${LOG_DIR:-logs}"
PID_DIR="${PID_DIR:-pids}"
mkdir -p "${LOG_DIR}" "${PID_DIR}" outputs

PIPELINE_LOG="${LOG_DIR}/${RUN_NAME}_pipeline.log"
PID_PATH="${PID_DIR}/${RUN_NAME}_pipeline.pid"
DATA_YAML="${DATASET_DIR}/data.yaml"
RUN_DIR="outputs/runs/${RUN_NAME}"
WEIGHTS="${RUN_DIR}/weights/best.pt"
SUBMIT_TXT_DIR="outputs/submission_${RUN_NAME}_txt"
SUBMIT_ZIP="outputs/submission_${RUN_NAME}.zip"
CHECKED_ZIP="outputs/submission_${RUN_NAME}_official_checked.zip"
INFERENCE_WORK_DIR="outputs/inference_inputs/${FUSION}"

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

ensure_inside_project() {
  local target
  target="$(realpath "$1")"
  case "${target}" in
    "$(pwd)"/datasets/*|"$(pwd)"/outputs/*)
      return 0
      ;;
    *)
      echo "Refusing to modify path outside generated project dirs: ${target}" >&2
      exit 1
      ;;
  esac
}

already_running() {
  pgrep -af "scripts/train_yolo.py" | grep -F -- "--name ${RUN_NAME}" >/dev/null 2>&1
}

main() {
  echo "$$" > "${PID_PATH}"
  log "pipeline started: ${RUN_NAME}"
  log "cwd=$(pwd)"
  log "raw_train=${RAW_TRAIN}"
  log "raw_test=${RAW_TEST}"
  log "dataset=${DATASET_DIR}"
  log "fusion=${FUSION}"
  log "modality_dropout=${MODALITY_DROPOUT}"
  log "reserve=${GPU_RESERVE_MB}MiB max_fraction=${MAX_BATCH_FRACTION}"

  if [[ ! -x ".venv/bin/python" ]]; then
    echo "Missing .venv/bin/python. Run remote setup first." >&2
    exit 1
  fi
  if [[ ! -f "${MODEL}" ]]; then
    echo "Missing model checkpoint: ${MODEL}" >&2
    exit 1
  fi
  if already_running; then
    echo "Run is already active: ${RUN_NAME}" >&2
    pgrep -af "scripts/train_yolo.py" | grep -F -- "--name ${RUN_NAME}" >&2 || true
    exit 1
  fi

  if command -v nvidia-smi >/dev/null; then
    log "initial GPU snapshot"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  fi

  if [[ "${FORCE_PREPARE}" == "1" && -e "${DATASET_DIR}" ]]; then
    ensure_inside_project "${DATASET_DIR}"
    log "removing generated dataset for FORCE_PREPARE=1: ${DATASET_DIR}"
    rm -rf "${DATASET_DIR}"
  fi

  if [[ ! -f "${DATA_YAML}" ]]; then
    log "preparing ${FUSION} dataset"
    prepare_args=(
      .venv/bin/python scripts/prepare_dataset.py
      --raw-root "${RAW_TRAIN}"
      --out-root "${DATASET_DIR}"
      --fusion "${FUSION}"
      --workers "${WORKERS}"
    )
    if [[ "${MODALITY_DROPOUT}" == "1" ]]; then
      prepare_args+=(--modality-dropout)
    fi
    "${prepare_args[@]}"
  else
    log "dataset exists, skipping prepare: ${DATA_YAML}"
  fi

  log "starting training"
  .venv/bin/python scripts/train_yolo.py \
    --data "${DATA_YAML}" \
    --model "${MODEL}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch auto-free \
    --gpu-reserve-mb "${GPU_RESERVE_MB}" \
    --max-batch-fraction "${MAX_BATCH_FRACTION}" \
    --workers "${WORKERS}" \
    --device "${DEVICE}" \
    --project outputs/runs \
    --name "${RUN_NAME}" \
    --patience "${PATIENCE}" \
    --amp \
    --no-plots

  if [[ ! -f "${WEIGHTS}" ]]; then
    echo "Training finished but best weights are missing: ${WEIGHTS}" >&2
    exit 1
  fi

  log "generating submission"
  ensure_inside_project "${SUBMIT_TXT_DIR}"
  ensure_inside_project "${INFERENCE_WORK_DIR}"
  rm -rf "${SUBMIT_TXT_DIR}" "${SUBMIT_ZIP}" "${CHECKED_ZIP}" "${INFERENCE_WORK_DIR}"
  .venv/bin/python scripts/predict_submit.py \
    --weights "${WEIGHTS}" \
    --raw-root "${RAW_TEST}" \
    --fusion "${FUSION}" \
    --work-dir outputs/inference_inputs \
    --out-dir "${SUBMIT_TXT_DIR}" \
    --zip-path "${SUBMIT_ZIP}" \
    --imgsz "${IMGSZ}" \
    --conf "${CONF}" \
    --iou "${IOU}" \
    --max-det "${MAX_DET}" \
    --device "${DEVICE}"

  log "repacking checked flat zip"
  .venv/bin/python scripts/repack_submission.py \
    --submission "${SUBMIT_ZIP}" \
    --raw-root "${RAW_TEST}" \
    --out-zip "${CHECKED_ZIP}" \
    --max-det "${MAX_DET}" \
    --num-classes 12

  .venv/bin/python scripts/validate_submission.py \
    --submission "${CHECKED_ZIP}" \
    --raw-root "${RAW_TEST}" \
    --max-det "${MAX_DET}" \
    --num-classes 12

  log "pipeline complete: ${CHECKED_ZIP}"
}

main "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
