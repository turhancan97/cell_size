#!/bin/bash
#SBATCH -p dgxa100 # partition (queue)
#SBATCH --gpus=1
#SBATCH --qos=big
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=segment_cells
#SBATCH --time=72:00:00 # time (D-HH:MM)
#SBATCH --output=logs/segment_cells_%j.log

export HYDRA_FULL_ERROR=1

nvidia-smi -L

conda init bash
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate cell-size

cell-size \
  data.data_dir=/shared/sets/datasets/vision/cellpose/Adults_training \
  segmentation.resize=1000 \
  output.generate_overlays=true \
  output.compute_cell_areas=true \
  output.generate_plots=true \
  cell_type=Adults_FrogBlood