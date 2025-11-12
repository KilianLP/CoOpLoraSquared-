#!/bin/bash
#SBATCH --job-name=lorasq_runs
#SBATCH --gres=gpu:1                
#SBATCH --partition=Brain3080       
#SBATCH --cpus-per-gpu=4            
#SBATCH --qos=highbrain
#SBATCH --output=logs/lorasq_runs_%j.out
#SBATCH --error=logs/lorasq_runs_%j.err

source /homes/k23preus/UE_Reherche/CLIP-LoRA/venv/bin/activate

EVAL_MODES=(avg_experts shared)

cd CoOpLoraSquared-
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OXFORD_FLOWERS_ROOT="/Brain/private/k23preus/data"
EUROSAT_ROOT="/homes/k23preus/data"
DTD_ROOT="/homes/k23preus/data"
OXFORD_PETS_ROOT="/Brain/private/k23preus/data"

for eval_mode in "${EVAL_MODES[@]}"; do
  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset oxford_flowers \
    --root_path "${OXFORD_FLOWERS_ROOT}" \
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
    --n_iters 20 \
    --exp_name "oxford_flowers_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"

  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset eurosat \
    --root_path "${EUROSAT_ROOT}" \
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
    --n_iters 30 \
    --exp_name "eurosat_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"

  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset dtd \
    --root_path "${DTD_ROOT}" \
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
    --n_iters 30 \
    --exp_name "dtd_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"

  python main.py \
    --mode lorasquared \
    --setting base2new \
    --dataset oxford_pets \
    --root_path "${OXFORD_PETS_ROOT}" \
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
    --n_iters 30 \
    --exp_name "oxford_pets_${eval_mode}" \
    --validate \
    --lorasquared_base_eval "${eval_mode}"
done
