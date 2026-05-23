#!/usr/bin/env bash

export DATA_DIR=/shared/sets/datasets/vision/cellpose/adults_second_trial
export FEEDBACK_CSVS='["/home/kargin/Projects/repositories/cell_size/latest.csv"]'
export OUTPUT_DIR=/home/kargin/Projects/repositories/cell_size/classifier_output

# Model and head search space
ENCODERS="resnet18,efficientnet_b0,squeezenet1_1"
TRAIN_WITH_VAL_MODE="false"         # both|true|false
USE_MLP_HEAD_MODE="false"           # both|true|false
USE_EFFICIENT_PROBING_MODE="false"  # both|true|false

bash scripts/launch_classifier_train_parallel.sh --max-concurrent 200 \
  --encoders "$ENCODERS" --freeze both \
  --train-with-val "$TRAIN_WITH_VAL_MODE" \
  --use-mlp-head "$USE_MLP_HEAD_MODE" \
  --use-efficient-probing "$USE_EFFICIENT_PROBING_MODE" \
  --lrs 0.0001 --thresholds 0.5 --cv off \
  --mask-background false \
  classifier.epochs=100 classifier.batch_size=64
