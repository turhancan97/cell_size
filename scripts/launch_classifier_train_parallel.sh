#!/usr/bin/env bash

set -u -o pipefail

# Submit classifier training runs to SLURM in parallel (one job per run, one GPU per job).
#
# Required env vars:
#   DATA_DIR=/path/to/segmented/data
#   FEEDBACK_CSVS='["/path/to/feedback.csv"]'
#   OUTPUT_DIR=/path/to/output_dir
#
# Usage:
#   bash scripts/launch_classifier_train_parallel.sh [--dry-run] [--max-concurrent N] \
#     [--encoders all|resnet18,resnet50,vit_b_16,efficientnet_b0,squeezenet1_1,timm/<model_name>] \
#     [--freeze both|true|false] \
#     [--train-with-val both|true|false] \
#     [--use-mlp-head both|true|false] \
#     [--use-efficient-probing both|true|false] \
#     [--lrs 0.001,0.0005] \
#     [--thresholds 0.7,0.8] \
#     [--cv both|on|off] \
#     -- [additional hydra overrides...]
#
# Example:
#   DATA_DIR=/data/segmented \
#   FEEDBACK_CSVS='["/data/feedback.csv"]' \
#   OUTPUT_DIR=/runs/cell_quality \
#   bash scripts/launch_classifier_train_parallel.sh --dry-run --max-concurrent 10 \
#     --encoders resnet18,resnet50,vit_b_16,efficientnet_b0,squeezenet1_1,timm/<model_name> --freeze both \
#     --train-with-val false \
#     --use-mlp-head both --use-efficient-probing false \
#     --lrs 0.001,0.0005 --thresholds 0.7 --cv off \
#     -- classifier.epochs=30 classifier.batch_size=64
#
# SLURM defaults (override via env vars):
#   SLURM_PARTITION=rtx4090_batch
#   SLURM_QOS=batch
#   SLURM_CPUS=10
#   SLURM_MEM=64G
#   SLURM_TIME=24:00:00
#   SLURM_ACCOUNT=<optional>
#
# Conda activation (override via env vars):
#   CONDA_SH=/path/to/conda.sh
#   CONDA_ENV=cell-size

DRY_RUN=0
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
ENCODERS_MODE="all"
FREEZE_MODE="both"
TRAIN_WITH_VAL_MODE="false"
MLP_HEAD_MODE="false"
EFFICIENT_PROBING_MODE="false"
LRS_CSV="0.001"
THRESHOLDS_CSV="0.7"
CV_MODE="off"
MASK_BG_MODE="false"
EXTRA_ARGS=()

CONDA_SH="${CONDA_SH:-/shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cell-size}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --max-concurrent)
      MAX_CONCURRENT="${2:-}"
      shift 2
      ;;
    --encoders)
      ENCODERS_MODE="${2:-}"
      shift 2
      ;;
    --freeze)
      FREEZE_MODE="${2:-}"
      shift 2
      ;;
    --train-with-val)
      TRAIN_WITH_VAL_MODE="${2:-}"
      shift 2
      ;;
    --use-mlp-head)
      MLP_HEAD_MODE="${2:-}"
      shift 2
      ;;
    --use-efficient-probing)
      EFFICIENT_PROBING_MODE="${2:-}"
      shift 2
      ;;
    --lrs)
      LRS_CSV="${2:-}"
      shift 2
      ;;
    --thresholds)
      THRESHOLDS_CSV="${2:-}"
      shift 2
      ;;
    --cv)
      CV_MODE="${2:-}"
      shift 2
      ;;
    --mask-background)
      MASK_BG_MODE="${2:-}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${DATA_DIR:-}" ]]; then
  echo "Missing required env var: DATA_DIR"
  exit 2
fi
if [[ -z "${FEEDBACK_CSVS:-}" ]]; then
  echo "Missing required env var: FEEDBACK_CSVS (JSON list string)"
  exit 2
fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "Missing required env var: OUTPUT_DIR"
  exit 2
fi

if ! [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT" -le 0 ]]; then
  echo "Invalid --max-concurrent value: '$MAX_CONCURRENT' (expected positive integer)"
  exit 2
fi

if [[ "$FREEZE_MODE" != "both" && "$FREEZE_MODE" != "true" && "$FREEZE_MODE" != "false" ]]; then
  echo "Invalid --freeze value: '$FREEZE_MODE' (expected: both|true|false)"
  exit 2
fi
if [[ "$TRAIN_WITH_VAL_MODE" != "both" && "$TRAIN_WITH_VAL_MODE" != "true" && "$TRAIN_WITH_VAL_MODE" != "false" ]]; then
  echo "Invalid --train-with-val value: '$TRAIN_WITH_VAL_MODE' (expected: both|true|false)"
  exit 2
fi
if [[ "$MLP_HEAD_MODE" != "both" && "$MLP_HEAD_MODE" != "true" && "$MLP_HEAD_MODE" != "false" ]]; then
  echo "Invalid --use-mlp-head value: '$MLP_HEAD_MODE' (expected: both|true|false)"
  exit 2
fi
if [[ "$EFFICIENT_PROBING_MODE" != "both" && "$EFFICIENT_PROBING_MODE" != "true" && "$EFFICIENT_PROBING_MODE" != "false" ]]; then
  echo "Invalid --use-efficient-probing value: '$EFFICIENT_PROBING_MODE' (expected: both|true|false)"
  exit 2
fi
if [[ "$CV_MODE" != "both" && "$CV_MODE" != "on" && "$CV_MODE" != "off" ]]; then
  echo "Invalid --cv value: '$CV_MODE' (expected: both|on|off)"
  exit 2
fi
if [[ "$MASK_BG_MODE" != "both" && "$MASK_BG_MODE" != "true" && "$MASK_BG_MODE" != "false" ]]; then
  echo "Invalid --mask-background value: '$MASK_BG_MODE' (expected: both|true|false)"
  exit 2
fi

split_csv() {
  local IFS=,
  read -r -a _arr <<< "$1"
  printf '%s\n' "${_arr[@]}"
}

PARTITION="${SLURM_PARTITION:-rtx4090_batch}"
QOS="${SLURM_QOS:-batch}"
CPUS="${SLURM_CPUS:-10}"
MEM="${SLURM_MEM:-64G}"
WALLTIME="${SLURM_TIME:-24:00:00}"
ACCOUNT="${SLURM_ACCOUNT:-}"

LOG_DIR="scripts/logs"
mkdir -p "$LOG_DIR"

if [[ $DRY_RUN -eq 0 && ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found: $CONDA_SH"
  exit 2
fi
if [[ $DRY_RUN -eq 0 ]] && ! command -v squeue >/dev/null 2>&1; then
  echo "squeue command not found; cannot enforce --max-concurrent limit."
  exit 2
fi

wait_for_slot() {
  local current_jobs
  while true; do
    current_jobs="$(squeue -h -u "$USER" -t R,PD | wc -l)"
    if [[ "$current_jobs" -lt "$MAX_CONCURRENT" ]]; then
      return 0
    fi
    echo "[THROTTLE] user_jobs=${current_jobs} >= max_concurrent=${MAX_CONCURRENT}; waiting 30s..."
    sleep 30
  done
}

TS="$(date +%Y%m%d_%H%M%S)"

if [[ "$ENCODERS_MODE" == "all" ]]; then
  ENCODERS=("resnet18" "resnet50" "vit_b_16" "efficientnet_b0" "squeezenet1_1" "timm/vit_small_patch16_dinov3.lvd1689m")
else
  mapfile -t ENCODERS < <(split_csv "$ENCODERS_MODE")
fi

if [[ "$FREEZE_MODE" == "both" ]]; then
  FREEZES=("true" "false")
else
  FREEZES=("$FREEZE_MODE")
fi

if [[ "$TRAIN_WITH_VAL_MODE" == "both" ]]; then
  TRAIN_WITH_VALS=("false" "true")
else
  TRAIN_WITH_VALS=("$TRAIN_WITH_VAL_MODE")
fi

if [[ "$MLP_HEAD_MODE" == "both" ]]; then
  MLP_HEADS=("false" "true")
else
  MLP_HEADS=("$MLP_HEAD_MODE")
fi

if [[ "$EFFICIENT_PROBING_MODE" == "both" ]]; then
  EFFICIENT_PROBINGS=("false" "true")
else
  EFFICIENT_PROBINGS=("$EFFICIENT_PROBING_MODE")
fi

mapfile -t LRS < <(split_csv "$LRS_CSV")
mapfile -t THRESHOLDS < <(split_csv "$THRESHOLDS_CSV")

if [[ "$CV_MODE" == "both" ]]; then
  CVS=("on" "off")
elif [[ "$CV_MODE" == "on" ]]; then
  CVS=("on")
else
  CVS=("off")
fi

if [[ "$MASK_BG_MODE" == "both" ]]; then
  MASK_BGS=("true" "false")
else
  MASK_BGS=("$MASK_BG_MODE")
fi

SUBMITTED=0
PLANNED=0

for ENC in "${ENCODERS[@]}"; do
  for FRZ in "${FREEZES[@]}"; do
    for TRV in "${TRAIN_WITH_VALS[@]}"; do
      for MLP in "${MLP_HEADS[@]}"; do
        for EFF in "${EFFICIENT_PROBINGS[@]}"; do
          if [[ "$MLP" == "true" && "$EFF" == "true" ]]; then
            echo "Skipping invalid combo for encoder=${ENC}: use_mlp_head=true and use_efficient_probing=true are mutually exclusive."
            continue
          fi
          if [[ "$EFF" == "true" ]]; then
            ENC_LOWER="$(echo "$ENC" | tr '[:upper:]' '[:lower:]')"
            if [[ "$ENC" != timm/* || "$ENC_LOWER" != *"vit"* ]]; then
              echo "Skipping invalid combo for encoder=${ENC}: use_efficient_probing=true requires timm ViT encoder."
              continue
            fi
          fi

          for LR in "${LRS[@]}"; do
            for THR in "${THRESHOLDS[@]}"; do
              for CV in "${CVS[@]}"; do
                if [[ "$TRV" == "true" && "$CV" == "on" ]]; then
                  echo "Skipping invalid combo: train_with_val=true is incompatible with cross-validation (encoder=${ENC})."
                  continue
                fi

                CV_FLAG="cv0"
                CV_OVERRIDES=("classifier.cross_validation.enabled=false")
                if [[ "$CV" == "on" ]]; then
                  CV_FLAG="cv1"
                  CV_OVERRIDES=("classifier.cross_validation.enabled=true")
                fi

                for MBG in "${MASK_BGS[@]}"; do
                  PLANNED=$((PLANNED+1))
                  # Deterministic run name. Timestamp ensures uniqueness per launcher invocation.
                  ENC_TAG="${ENC//\//_}"
                  RUN_NAME="clf_${ENC_TAG}_freeze${FRZ}_trv${TRV}_mlp${MLP}_eff${EFF}_lr${LR}_thr${THR}_maskbg${MBG}_${CV_FLAG}_${TS}"
                  RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
                  BEST_PATH="${RUN_DIR}/best_model.pt"

                  if [[ -f "$BEST_PATH" ]]; then
                    echo "Skipping ${RUN_NAME} (best model exists: ${BEST_PATH})"
                    continue
                  fi

                  JOB_NAME="cellsize_${RUN_NAME}"
                  OUT_FILE="${LOG_DIR}/${JOB_NAME}_%j_${TS}.log"

                  TRAIN_ARGS=(
                    "feedback_csvs=${FEEDBACK_CSVS}"
                    "data_dir=${DATA_DIR}"
                    "output_dir=${OUTPUT_DIR}"
                    "classifier.encoder=${ENC}"
                    "classifier.freeze_encoder=${FRZ}"
                    "classifier.train_with_val=${TRV}"
                    "classifier.use_mlp_head=${MLP}"
                    "classifier.use_efficient_probing=${EFF}"
                    "classifier.learning_rate=${LR}"
                    "classifier.confidence_threshold=${THR}"
                    "classifier.mask_background=${MBG}"
                    "classifier.wandb.enabled=false"
                    "classifier.wandb.run_name=${RUN_NAME}"
                    "${CV_OVERRIDES[@]}"
                    "${EXTRA_ARGS[@]}"
                  )
                  TRAIN_ARGS_STR="${TRAIN_ARGS[*]}"
                  WRAP_CMD="bash -lc 'source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cell-size-train ${TRAIN_ARGS_STR}'"

                  SBATCH_CMD=(
                    sbatch
                    --partition="$PARTITION"
                    --qos="$QOS"
                    --gpus=1
                    --cpus-per-task="$CPUS"
                    --mem="$MEM"
                    --ntasks=1
                    --time="$WALLTIME"
                    --job-name="$JOB_NAME"
                    --output="$OUT_FILE"
                  )
                  if [[ -n "$ACCOUNT" ]]; then
                    SBATCH_CMD+=(--account="$ACCOUNT")
                  fi
                  SBATCH_CMD+=(--wrap "$WRAP_CMD")

                  echo "Submitting ${RUN_NAME}"
                  echo "Log: ${OUT_FILE}"
                  echo "Cmd: ${SBATCH_CMD[*]}"

                  if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[DRY-RUN] Skipping submission"
                    echo
                    continue
                  fi

                  wait_for_slot
                  "${SBATCH_CMD[@]}"
                  RC=$?
                  if [[ $RC -eq 0 ]]; then
                    SUBMITTED=$((SUBMITTED+1))
                  else
                    echo "[ERROR] Submission failed for ${RUN_NAME} (rc=${RC})"
                    exit 2
                  fi
                  echo
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "Parallel launch summary: planned=${PLANNED}, submitted=${SUBMITTED}, max_concurrent=${MAX_CONCURRENT}, dry_run=${DRY_RUN}"
