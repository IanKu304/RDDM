#!/usr/bin/env bash
set -euo pipefail

# RDDM Experiment 2 (Deraining) - Ablation runner (Linux)
# 需要你先在 train.py 支援下列參數：
#   --run_name --steps --seed --amp --beta_end --beta_scale --sampling_timesteps
#
# 用法：
#   conda activate rddm
#   cd experiments/2_Image_Restoration_deraing_raindrop_noise1
#   bash run_grid_linux.sh 40000   # stage1
#   bash run_grid_linux.sh 120000  # stage2（只挑選少數組時請自行改下面的列表）

STEPS="${1:-40000}"
SEED="${SEED:-10}"
SAMP="${SAMP:-10}"

# 3×3 grid
BETA_END_LIST=(0.01 0.02 0.04)
BETA_SCALE_LIST=(0.5 1.0 2.0)

for be in "${BETA_END_LIST[@]}"; do
  for bs in "${BETA_SCALE_LIST[@]}"; do
    run_name="GT-RAIN__img256__bs1__acc2__steps${STEPS}__schedLinear__betaEnd${be}__betaScale${bs}__seed${SEED}"
    echo "[INFO] Run: ${run_name}"
    python train.py       --sampling_timesteps "${SAMP}"       --steps "${STEPS}"       --seed "${SEED}"       --amp       --beta_end "${be}"       --beta_scale "${bs}"       --run_name "${run_name}"
  done
done
