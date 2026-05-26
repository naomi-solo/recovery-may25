#!/usr/bin/env python3
"""
Minimal cosine-only verification of PCA component alignment.

For each dataset, this script:
  1) loads a dataset-specific PCA artifact
  2) pulls perturbation vectors from stored records
  3) computes cosine alignment to one selected component direction
  4) writes summary CSV + histogram

This intentionally avoids automatic component-selection heuristics.
Pick a component index explicitly (default: 0).

Example:
  python analysis/verify_component_alignment.py \
    --datasets gsm_mc,math_mc,mmlu,bbq,sgxs \
    --pca_dir archive/pca_from_sweep \
    --eps_tag 0.1 \
    --regime shared_flip \
    --mode chosen \
    --component_idx 0 \
    --outdir results/analysis/component_alignment
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde


DEFAULT_DATASETS = ["gsm_mc", "math_mc", "mmlu", "bbq", "sgxs"]


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def filter_vectors(records: List[dict], mode: str) -> np.ndarray:
    vecs = []
    for r in records:
        ctype = r.get("completion_type", None)
        if mode in ("chosen", "rejected") and ctype != mode:
            continue
        v = r.get("perturbation_direction", None)
        if isinstance(v, torch.Tensor) and v.ndim == 1:
            vecs.append(v.float())

    if not vecs:
        raise RuntimeError(f"No perturbation vectors after mode={mode} filter.")
    return torch.stack(vecs, dim=0).cpu().numpy()


def cosine_alignment(X: np.ndarray, u_unit: np.ndarray) -> np.ndarray:
    Xn = normalize_rows(X)
    return Xn @ u_unit


def _auto_hist_bins(values: np.ndarray) -> int:
    """
    Freedman–Diaconis style bins, with sensible clamps.
    This is usually more stable than fixed or sqrt bins when shapes differ by dataset.
    """
    n = max(1, int(values.size))
    if n < 3:
        return 10

    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr <= 1e-12:
        return int(np.clip(np.sqrt(n), 8, 25))

    h = 2.0 * iqr / (n ** (1.0 / 3.0))
    if h <= 1e-12:
        return int(np.clip(np.sqrt(n), 8, 25))

    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if data_max <= data_min:
        return 10

    bins = int(np.ceil((data_max - data_min) / h))
    return int(np.clip(bins, 8, 35))


def _auto_xlim(values: np.ndarray) -> tuple[float, float]:
    # Trim extreme tails to make the central distribution readable.
    lo = float(np.quantile(values, 0.01))
    hi = float(np.quantile(values, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return -1.0, 1.0

    pad = 0.08 * (hi - lo)
    lo = max(-1.0, lo - pad)
    hi = min(1.0, hi + pad)
    return lo, hi


def save_hist(values: np.ndarray, title: str, out_path: Path, full_range: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = _auto_hist_bins(values)

    mu = float(np.mean(values))
    med = float(np.median(values))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Left: histogram + KDE for shape
    ax0.hist(values, bins=bins, alpha=0.75, density=True, edgecolor="white", linewidth=0.6)
    try:
        kde = gaussian_kde(values)
        xs = np.linspace(np.min(values), np.max(values), 400)
        ax0.plot(xs, kde(xs), linewidth=1.5, color="tab:orange", label="kde")
    except Exception:
        # If KDE fails (e.g., near-zero variance), silently keep histogram-only.
        pass

    # Helpful reference lines
    ax0.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7, color="black", label="0")
    ax0.axvline(mu, linestyle="-", linewidth=1.2, alpha=0.9, color="tab:red", label=f"mean={mu:+.3f}")
    ax0.axvline(med, linestyle=":", linewidth=1.4, alpha=0.9, color="tab:green", label=f"median={med:+.3f}")

    if full_range:
        xlo, xhi = -1.0, 1.0
    else:
        xlo, xhi = _auto_xlim(values)
    ax0.set_xlim(xlo, xhi)
    ax0.set_title(title)
    ax0.set_xlabel("cosine(x, component)")
    ax0.set_ylabel("density")
    ax0.legend(fontsize=8)

    # Right: ECDF to show distribution shape without bin sensitivity
    x_sorted = np.sort(values)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    ax1.plot(x_sorted, y, linewidth=1.6, color="tab:blue")
    ax1.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7, color="black")
    ax1.axvline(mu, linestyle="-", linewidth=1.2, alpha=0.9, color="tab:red")
    ax1.axvline(med, linestyle=":", linewidth=1.4, alpha=0.9, color="tab:green")
    ax1.set_xlim(xlo, xhi)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("ECDF")
    ax1.set_xlabel("cosine(x, component)")
    ax1.set_ylabel("fraction ≤ x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--pca_dir", default="archive/pca_from_sweep")
    ap.add_argument("--eps_tag", default="0.1")
    ap.add_argument("--regime", default="shared_flip")
    ap.add_argument("--mode", choices=["chosen", "rejected", "both"], default="chosen")
    ap.add_argument("--component_idx", type=int, default=0)
    ap.add_argument("--full_range", action="store_true", help="Use full x-range [-1, 1] instead of auto zoom.")
    ap.add_argument("--outdir", default="results/analysis/component_alignment")
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    pca_dir = Path(args.pca_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for ds in datasets:
        pca_path = pca_dir / f"pca_{ds}_eps{args.eps_tag}_{args.regime}_k20.pt"
        if not pca_path.exists():
            print(f"[skip] missing pca: {pca_path}")
            continue

        obj = torch.load(pca_path, map_location="cpu")
        comps = obj["components"].float().cpu().numpy()  # (K, D)
        evr = obj.get("explained_var_ratio", None)
        records = obj.get("records", [])

        if not records:
            print(f"[skip] no records in {pca_path}")
            continue

        k = int(args.component_idx)
        if k < 0 or k >= comps.shape[0]:
            print(f"[skip] component_idx={k} out of range for {pca_path} (K={comps.shape[0]})")
            continue

        X = filter_vectors(records, mode=args.mode)
        u = normalize_vec(comps[k])
        cos = cosine_alignment(X, u)

        save_hist(
            cos,
            title=f"{ds} | component {k} | cosine alignment",
            out_path=outdir / "figs" / f"{ds}__component{k}__cosine_hist.png",
            full_range=args.full_range,
        )

        row = {
            "dataset": ds,
            "component": k,
            "mode": args.mode,
            "n_vectors": int(len(cos)),
            "cos_mean": float(np.mean(cos)),
            "cos_median": float(np.median(cos)),
            "cos_p25": float(np.quantile(cos, 0.25)),
            "cos_p75": float(np.quantile(cos, 0.75)),
            "pca_path": str(pca_path),
        }

        if isinstance(evr, torch.Tensor) and 0 <= k < evr.shape[0]:
            row["evr"] = float(evr[k].item())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = outdir / "summary_cosine_only.csv"
    summary.to_csv(summary_path, index=False)

    print(f"wrote: {summary_path}")
    if len(summary):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()