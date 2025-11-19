#!/bin/bash
#SBATCH --job-name=lorasq_runs_2
#SBATCH --gres=gpu:1
#SBATCH --partition=Brain3080
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=highbrain
#SBATCH --output=logs/lorasq_runs_2.out
#SBATCH --error=logs/lorasq_runs_2.err

source /homes/k23preus/UE_Reherche/CLIP-LoRA/venv/bin/activate

EVAL_MODES=(avg_experts shared)

cd CoOpLoraSquared-
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CALTECH_ROOT="/Brain/private/k23preus/data"
UCF101_ROOT="/Brain/private/k23preus/data"
SUN397_ROOT="/Brain/private/k23preus/data"

for eval_mode in "${EVAL_MODES[@]}"; do
  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset caltech101 \
    --root_path "${CALTECH_ROOT}" \
    --shots 16 \
    --backbone ViT-B/16 \
    --encoder both \
    --position all \
    --params q k v \
    --lora_shared_rank 4 \
    --lora_expert_rank 2 \
    --lr 2e-4 \
    --batch_size 32 \
    --test_batch_size 32 \
    --n_iters 40 \
    --exp_name "caltech101_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"

  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset ucf101 \
    --root_path "${UCF101_ROOT}" \
    --shots 16 \
    --backbone ViT-B/16 \
    --encoder both \
    --position all \
    --params q k v \
    --lora_shared_rank 4 \
    --lora_expert_rank 2 \
    --lr 2e-4 \
    --batch_size 32 \
    --test_batch_size 32 \
    --n_iters 40 \
    --exp_name "ucf101_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"

  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset sun397 \
    --root_path "${SUN397_ROOT}" \
    --shots 16 \
    --backbone ViT-B/16 \
    --encoder both \
    --position all \
    --params q k v \
    --lora_shared_rank 4 \
    --lora_expert_rank 2 \
    --lr 2e-4 \
    --batch_size 32 \
    --test_batch_size 32 \
    --n_iters 40 \
    --exp_name "sun397_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"
done
