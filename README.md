# CoOpLoRA-Squared

Research project based on the paper **" Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages " (CVPR 2025)**.

This project implements a new method, LoRA-Squared, which combines a shared LoRA module across all classes with expert LoRA modules assigned to each specific class. During evaluation on unseen classes, only the shared LoRA module is used.


## 1. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r deps/requirements.txt
```

## 2. Datasets
1. Download the raw datasets you need (Oxford Pets, DTD, Food101, …) and unpack each one under a single root folder, e.g. `/data`.
2. Fetch the public few-shot splits and place them alongside the datasets:
   ```bash
   export DATA_PATH=/data
   bash scripts/download_splits.sh
   ```
   Each dataset folder will now contain `split_fewshot/shot_<k>-seed_<s>_{train,val}.jsonl`.

## 3. Running experiments
All entry points share the same CLI (`python main.py --help`). Example LoRA² base→novel run:
```bash
python main.py \
  --mode lorasquared \
  --setting base2new \
  --dataset dtd \
  --root_path /data \
  --shots 4 \
  --backbone ViT-B/16 \
  --encoder both \
  --position all \
  --params q k v \
  --lora_shared_rank 4 \
  --lora_expert_rank 2 \
  --batch_size 32 \
  --test_batch_size 32 \
  --n_iters 300 \
  --lr 2e-4 \
  --exp_name dtd_lorasq_b2n
```
Results are written to `results/<setting>/<backbone>/<dataset>/shots_<k>/seed_<s>/.../<exp_name>.csv`.

## 4. Tips
- Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` if you keep deterministic CUDA enabled (default in `main.py`).
- Reduce `--workers` if the dataloader warns about too many worker processes.
- Combine LoRA² experts at test time with `--lorasquared_base_eval shared|avg_experts` to switch between shared-only or averaged expert routing for base classes.
- Use `summarize.py` to average multiple seeds after a sweep.

