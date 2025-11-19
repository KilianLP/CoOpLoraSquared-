#!/bin/bash
#SBATCH --job-name=cliplora_all
#SBATCH --gres=gpu:1
#SBATCH --partition=Brain3080
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=highbrain
#SBATCH --output=logs/cliplora_all.out
#SBATCH --error=logs/cliplora_all.err

source /homes/k23preus/UE_Reherche/CLIP-LoRA/venv/bin/activate

cd CoOpLoraSquared-
export CUBLAS_WORKSPACE_CONFIG=:4096:8

declare -A ROOTS=(
  [oxford_flowers]="/Brain/private/k23preus/data"
  [eurosat]="/Brain/private/k23preus/data"
  [dtd]="/Brain/private/k23preus/data"
  [oxford_pets]="/Brain/private/k23preus/data"
  [caltech101]="/Brain/public/datasets"
  [ucf101]="/Brain/public/datasets"
  [sun397]="/Brain/public/datasets"
)

for dataset in "${!ROOTS[@]}"; do
  python main.py \
    --mode cliplora \
    --setting base2new \
    --dataset "${dataset}" \
    --root_path "${ROOTS[$dataset]}" \
    --shots 16 \
    --backbone ViT-B/16 \
    --encoder both \
    --position all \
    --params q k v \
    --r 4 \
    --alpha 4 \
    --dropout_rate 0.25 \
    --lr 2e-4 \
    --batch_size 32 \
    --test_batch_size 32 \
    --n_iters 20 \
    --exp_name "cliplora_${dataset}" \
    --validate
done
