#!/bin/bash
#SBATCH -p dgxh100 # partition (queue)
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --job-name=classify_adults
#SBATCH --time=24:00:00 # time (D-HH:MM)
#SBATCH --output=logs/classify_adults_%j.log

export HYDRA_FULL_ERROR=1

nvidia-smi -L

conda init bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate cell-size

cell-size-classify \
    checkpoint=./classifier_output/run_2/best_model.pt \
    data_dir=/shared/sets/datasets/vision/cellpose/Adults_training \
    output_dir=./classify_output \
    classifier.confidence_threshold=0.5 \
    classifier.selective_rejection.enabled=true \
    classifier.selective_rejection.t_bad=0.10 \
    classifier.selective_rejection.t_good=0.76 \
    generate_filtered_overlays=false