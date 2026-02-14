"""
Compute cosine similarity (Frobenius inner product) between shared and expert LoRA^2
adapters saved by `save_lorasquared`, and plot aggregate statistics.

Assumptions about checkpoint structure:
- Each .pt file contains {"weights": {...}, "metadata": {...}}
- weights: layer_{i} -> {proj_name -> {lora_shared_A, lora_shared_B, lora_expert_A, lora_expert_B, ...}}
- Shapes follow LinearLoRASquared: shared update = B_shared @ A_shared; expert update = B_exp @ A_exp.

Outputs (written to --outdir, default plots_ortho):
- ortho_metrics.csv : per-(file,layer,proj,expert) cosine similarities.
- mean_overall.txt   : single overall mean cosine.
- Plots:
  * cosine_vs_depth.png          : mean±std over layers (averaged across files/projs/experts).
  * cosine_by_projection.png     : boxplot by projection (q/k/v/o).
  * heatmap_layer_expert.png     : heatmap of mean cosine (layer index x expert id).
  * cosine_by_dataset.png        : bar of mean cosine per dataset (if dataset inferred).
  * cosine_by_r_shared.png       : boxplot grouped by shared rank (if metadata present).
  * cosine_by_r_expert.png       : boxplot grouped by expert rank (if metadata present).

Usage:
  python scripts/orthogonality_lorasq.py \
      --root ./checkpoints \
      --outdir plots_ortho

Notes:
- Cosine uses |<S, E>| / (||S|| ||E||), where <.,.> is Frobenius inner product.
- If an expert or shared update has zero norm, that pair is skipped.
- Dataset/backbone/shots/seed/exp are inferred from the path when possible.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, default="./checkpoints", help="Root folder containing .pt LoRA^2 checkpoints.")
    p.add_argument("--outdir", type=str, default="plots_ortho", help="Where to save plots and CSV.")
    p.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern for checkpoint filenames.")
    p.add_argument(
        "--proj_filter",
        nargs="*",
        default=None,
        help="Optional list of projections to include (q k v o). Default: all present.",
    )
    return p.parse_args()


def infer_metadata(path: str) -> Dict[str, Optional[str]]:
    parts = Path(path).parts
    # Expect .../<root>/<backbone>/<dataset>/<shots>shots/seed<seed>/<file>.pt
    meta = {
        "backbone": None,
        "dataset": None,
        "shots": None,
        "seed": None,
        "file": Path(path).name,
        "r_shared": None,
        "r_expert": None,
    }
    try:
        meta["backbone"] = parts[-5]
        meta["dataset"] = parts[-4]
        shots_part = parts[-3]
        if shots_part.endswith("shots"):
            meta["shots"] = shots_part.replace("shots", "")
        seed_part = parts[-2]
        if seed_part.startswith("seed"):
            meta["seed"] = seed_part.replace("seed", "")
    except Exception:
        pass
    return meta


def lora_update(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # shapes: A [r, in], B [out, r]; update = B @ A
    return B @ A


def cosine_fro(shared: torch.Tensor, expert: torch.Tensor) -> Optional[float]:
    s = shared.flatten()
    e = expert.flatten()
    s_norm = torch.norm(s)
    e_norm = torch.norm(e)
    if s_norm == 0 or e_norm == 0:
        return None
    ip = torch.dot(s, e)
    cos = torch.abs(ip) / (s_norm * e_norm)
    return float(cos.item())


def extract_cosines(
    ckpt: dict, proj_filter: Optional[List[str]]
) -> List[Dict[str, object]]:
    records = []
    weights = ckpt.get("weights", {})
    for layer_name, proj_dict in weights.items():
        layer_idx = int(layer_name.replace("layer_", "")) if layer_name.startswith("layer_") else None
        for proj, tensors in proj_dict.items():
            if proj_filter and proj not in proj_filter:
                continue
            A_shared = tensors.get("lora_shared_A")
            B_shared = tensors.get("lora_shared_B")
            experts_A = tensors.get("lora_expert_A", [])
            experts_B = tensors.get("lora_expert_B", [])
            if A_shared is None or B_shared is None:
                continue
            shared_update = lora_update(A_shared, B_shared)
            for exp_id, (A_e, B_e) in enumerate(zip(experts_A, experts_B)):
                expert_update = lora_update(A_e, B_e)
                cos = cosine_fro(shared_update, expert_update)
                if cos is None:
                    continue
                records.append(
                    {
                        "layer": layer_idx,
                        "proj": proj,
                        "expert": exp_id,
                        "cosine": cos,
                    }
                )
    return records


def plot_cosine_vs_depth(df: pd.DataFrame, outdir: Path):
    if "layer" not in df.columns:
        return
    grouped = df.groupby("layer")["cosine"]
    mean = grouped.mean()
    std = grouped.std()
    plt.figure(figsize=(8, 4))
    plt.plot(mean.index, mean.values, marker="o", label="mean")
    plt.fill_between(mean.index, mean - std, mean + std, alpha=0.2, label="±1 std")
    plt.xlabel("Layer index (wrapped order)")
    plt.ylabel("Cosine |<S,E>|/(||S||·||E||)")
    plt.title("Shared vs Expert similarity by depth")
    plt.legend()
    out = outdir / "cosine_vs_depth.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_by_projection(df: pd.DataFrame, outdir: Path):
    if "proj" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    df.boxplot(column="cosine", by="proj")
    plt.suptitle("")
    plt.title("Cosine by projection")
    plt.ylabel("Cosine")
    out = outdir / "cosine_by_projection.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_mean_over_experts(df: pd.DataFrame, outdir: Path):
    if "layer" not in df.columns:
        return
    grouped = df.groupby("layer")["cosine"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(grouped.index, grouped.values, marker="o")
    plt.xlabel("Layer index")
    plt.ylabel("Mean cosine across experts")
    plt.title("Shared vs Expert similarity (experts averaged)")
    out = outdir / "cosine_vs_depth_mean_experts.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_by_dataset(df: pd.DataFrame, outdir: Path):
    if "dataset" not in df.columns or df["dataset"].isna().all():
        return
    grouped = df.groupby("dataset")["cosine"].mean().sort_values(ascending=False)
    plt.figure(figsize=(max(6, len(grouped) * 0.8), 4))
    grouped.plot(kind="bar")
    plt.ylabel("Mean cosine")
    plt.title("Mean cosine per dataset")
    out = outdir / "cosine_by_dataset.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_by_rank(df: pd.DataFrame, outdir: Path, rank_col: str, filename: str, title: str):
    if rank_col not in df.columns or df[rank_col].isna().all():
        return
    plt.figure(figsize=(6, 4))
    df.boxplot(column="cosine", by=rank_col)
    plt.suptitle("")
    plt.title(title)
    plt.ylabel("Cosine")
    out = outdir / filename
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True)
    if not files:
        print("No checkpoint files found; exiting.")
        return

    all_records: List[Dict[str, object]] = []
    for path in files:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as e:  # noqa: broad-except
            print(f"[skip] failed to load {path}: {e}")
            continue
        meta = infer_metadata(path)
        ckpt_meta = ckpt.get("metadata", {})
        meta["r_shared"] = ckpt_meta.get("r_shared", meta.get("r_shared"))
        meta["r_expert"] = ckpt_meta.get("r_expert", meta.get("r_expert"))
        recs = extract_cosines(ckpt, args.proj_filter)
        for r in recs:
            r.update(meta)
            r["path"] = path
        all_records.extend(recs)

    if not all_records:
        print("No cosine records computed; exiting.")
        return

    df = pd.DataFrame(all_records)
    csv_path = outdir / "ortho_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    overall_mean = df["cosine"].mean()
    with open(outdir / "mean_overall.txt", "w") as f:
        f.write(f"{overall_mean:.6f}\n")
    print(f"Overall mean cosine: {overall_mean:.4f}")

    plot_cosine_vs_depth(df, outdir)
    plot_by_projection(df, outdir)
    plot_mean_over_experts(df, outdir)
    plot_by_dataset(df, outdir)
    plot_by_rank(df, outdir, "r_shared", "cosine_by_r_shared.png", "Cosine by shared rank")
    plot_by_rank(df, outdir, "r_expert", "cosine_by_r_expert.png", "Cosine by expert rank")


if __name__ == "__main__":
    main()
