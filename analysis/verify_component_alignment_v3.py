#!/usr/bin/env python3
"""
Verify PCA component alignment using sweep-produced PCA artifacts.

What this script does per PCA file:
  1) load PCA artifact + records
  2) compute cosine alignment between each perturbation vector and one selected component
  3) select component either explicitly (--component_idx) or automatically from top-N components
     by maximum |corr(score_k, delta_margin)| where delta_margin = adv_margin - clean_margin
  4) save histogram+ECDF figure and write summary CSV

Works with sweep outputs under:
  results/outputs/pca/pca_seed*_layer*_eps*_k*_*_*_*_n*.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde

import json

LABELS_GLOB_DEFAULT = "results/outputs/global_verify_components/labels/component_labels_*.jsonl"
    
PCA_PAT_OLD = re.compile(
    r"pca_seed(?P<seed>\d+)_layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_k(?P<k>\d+)_(?P<cc>\w+)_(?P<tag>\w+)_(?P<mode>\w+)_n(?P<n>\d+)\.pt$"
)
PCA_PAT_NEW = re.compile(
    r"pca_(?P<dataset>.+?)_(?P<split>train|holdout)_seed(?P<seed>\d+)_"
    r"layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_k(?P<k>\d+)_"
    r"(?P<cc>ambig|disambig)_(?P<tag>shared_flip|shared_noflip|separate)_(?P<mode>chosen|rejected|both)_n(?P<n>\d+)\.pt$"
)
DIR_PAT = re.compile(
    r"direction_records_(?P<dataset>[^_]+(?:_[^_]+)?)_(?P<split>[^_]+)_seed\d+_layer\d+_eps[\d\.]+_\w+_\w+_n\d+\.pt$"
)

def _parse_pca_name(name: str) -> Optional[dict]:
    m_new = PCA_PAT_NEW.match(name)
    if m_new:
        d = m_new.groupdict()
        d["name_format"] = "new"
        return d

    m_old = PCA_PAT_OLD.match(name)
    if m_old:
        d = m_old.groupdict()
        d["name_format"] = "old"
        d["dataset"] = ""
        d["split"] = ""
        return d

    return None

def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def cosine_alignment(X: np.ndarray, u_unit: np.ndarray) -> np.ndarray:
    return normalize_rows(X) @ u_unit


def _auto_hist_bins(values: np.ndarray) -> int:
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
    lo = float(np.quantile(values, 0.01))
    hi = float(np.quantile(values, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return -1.0, 1.0
    pad = 0.08 * (hi - lo)
    return max(-1.0, lo - pad), min(1.0, hi + pad)


def save_hist(values: np.ndarray, title: str, out_path: Path, full_range: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = _auto_hist_bins(values)
    mu = float(np.mean(values))
    med = float(np.median(values))

    fig, ax0 = plt.subplots(1, 1, figsize=(9, 4.8))

    ax0.hist(values, bins=bins, alpha=0.75, density=True, edgecolor="white", linewidth=0.6)
    try:
        kde = gaussian_kde(values)
        xs = np.linspace(np.min(values), np.max(values), 400)
        ax0.plot(xs, kde(xs), linewidth=1.5, color="tab:orange", label="kde")
    except Exception:
        pass
    
    ax0.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7, color="black", label="0")
    ax0.axvline(mu, linestyle="-", linewidth=1.2, alpha=0.9, color="tab:red", label=f"mean={mu:+.3f}")
    ax0.axvline(med, linestyle=":", linewidth=1.4, alpha=0.9, color="tab:green", label=f"median={med:+.3f}")
    
    xlo, xhi = (-1.0, 1.0) if full_range else _auto_xlim(values)
    ax0.set_xlim(xlo, xhi)
    ax0.set_title(title, fontsize=11, wrap=True)
    ax0.set_xlabel("cosine(x, component)")
    ax0.set_ylabel("density")
    ax0.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    xs = x[ok]
    ys = y[ok]
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _dataset_from_obj(obj: dict, fallback: str) -> str:
    in_path = obj.get("in_path", "")
    if isinstance(in_path, str):
        m = DIR_PAT.search(Path(in_path).name)
        if m:
            return m.group("dataset")

    recs = obj.get("records", [])
    if recs:
        ds = recs[0].get("dataset", None)
        if isinstance(ds, str) and ds:
            return ds
    return fallback

def load_component_labels(labels_dir: str) -> dict:
    """
    Returns mapping:
      labels[(scope_name, cc, tag, mode)][component_idx] = "short label"
    """
    import re
    from pathlib import Path

    pat = re.compile(
        r"component_labels_(?P<scope>.+?)_(?P<split>train|holdout)_seed(?P<seed>\d+)_"
        r"layer(?P<layer>\d+)_eps(?P<eps>[\d\.]+)_K(?P<K>\d+)_"
        r"(?P<cc>\w+)_(?P<tag>shared_flip|shared_noflip|separate)_(?P<mode>\w+)_n(?P<n>\d+)\.jsonl$"
    )

    mapping = {}
    for path in Path(labels_dir).glob("component_labels_*.jsonl"):
        m = pat.match(path.name)
        if not m:
            continue

        scope_name = m.group("scope")
        cc = m.group("cc")
        tag = m.group("tag")
        mode = m.group("mode")

        key = (scope_name, cc, tag, mode)
        mapping.setdefault(key, {})

        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                comp = int(row["component"])
                mapping[key][comp] = row.get("label", "")

    return mapping
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pca_glob",
        default="results/outputs/global_verify_components/pca/pca*.pt"
    )
    ap.add_argument("--datasets", default="", help="comma list; empty means all")
    ap.add_argument("--tags", default="", help="comma list; empty means all")
    ap.add_argument("--modes", default="", help="comma list; empty means all")
    ap.add_argument("--ccs", default="", help="comma list; empty means all")
    ap.add_argument("--top_n", type=int, default=5, help="consider top-N components (by EVR) for auto-selection")
    ap.add_argument("--component_idx", type=int, default=-1, help="if >=0, use this component for every config")
    ap.add_argument("--full_range", action="store_true", help="Use full x-range [-1, 1] instead of auto zoom")
    ap.add_argument(
        "--labels_dir",
        default="results/outputs/global_verify_components/labels",
        help="Directory containing component_labels_*.jsonl files",
    )
    ap.add_argument("--outdir", default="results/analysis/global_topN_component_alignment_sweep")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label_map = load_component_labels(args.labels_dir)

    allowed_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}
    allowed_tags = {x.strip() for x in args.tags.split(",") if x.strip()}
    allowed_modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    allowed_ccs = {x.strip() for x in args.ccs.split(",") if x.strip()}

    rows = []
    files = sorted(Path().glob(args.pca_glob))
    if not files:
        raise SystemExit(f"No PCA files matched: {args.pca_glob}")

    for pca_path in files:
        meta = _parse_pca_name(pca_path.name)
        if meta is None:
            continue
        if allowed_tags and meta["tag"] not in allowed_tags:
            continue
        if allowed_modes and meta["mode"] not in allowed_modes:
            continue
        if allowed_ccs and meta["cc"] not in allowed_ccs:
            continue

        obj = torch.load(pca_path, map_location="cpu")
        dataset = meta["dataset"] if meta.get("dataset") else _dataset_from_obj(obj, fallback="unknown")
        perturbation_type = meta["tag"]
        if allowed_datasets and dataset not in allowed_datasets:
            continue

        records = obj.get("records", [])
        if not records:
            print(f"[skip] no records in {pca_path}")
            continue

        comps = obj["components"].float().cpu().numpy()  # (K, D)
        scores = obj.get("scores", None)
        if isinstance(scores, torch.Tensor):
            scores = scores.float().cpu().numpy()
        else:
            scores = None

        X = torch.stack([r["perturbation_direction"] for r in records], dim=0).float().cpu().numpy()

        delta_margin = np.array(
            [float(r.get("adv_margin", np.nan)) - float(r.get("clean_margin", np.nan)) for r in records],
            dtype=float,
        )

        if args.component_idx >= 0:
            selected_components = [int(args.component_idx)]
            reason = "explicit"
            top_considered = str(int(args.component_idx))
        else:
            n_top = max(1, min(int(args.top_n), comps.shape[0]))
            evr = obj.get("explained_var_ratio", None)
            if isinstance(evr, torch.Tensor):
                ev = evr.float().cpu().numpy()
                selected_components = np.argsort(-ev)[:n_top].tolist()
            else:
                selected_components = list(range(n_top))

            reason = f"top{n_top}_by_explained_variance"
            top_considered = ",".join(str(x) for x in selected_components)

        is_global = bool(obj.get("global_pca", False)) or dataset == "global"

        if is_global:
            datasets_in_records = sorted(
                {r.get("dataset", "unknown") for r in records if isinstance(r.get("dataset", "unknown"), str)}
            )
        else:
            datasets_in_records = [dataset]

        for k in selected_components:
            if k < 0 or k >= comps.shape[0]:
                print(f"[skip] component {k} out of range for {pca_path}")
                continue

            u = normalize_vec(comps[k])
            cos = np.abs(cosine_alignment(X, u))

            for ds in datasets_in_records:
                if allowed_datasets and ds not in allowed_datasets:
                    continue

                if is_global:
                    idx = [i for i, r in enumerate(records) if r.get("dataset", "unknown") == ds]
                else:
                    idx = list(range(len(records)))

                if not idx:
                    continue

                cos_ds = cos[idx]
                scores_ds = scores[idx, k] if scores is not None else None
                delta_margin_ds = delta_margin[idx]
                corr_k_ds = _corr(scores_ds, delta_margin_ds) if scores_ds is not None else float("nan")

                fig_rel = Path("figs") / (
                    f"{ds}__scope-{dataset}__ptype-{perturbation_type}__seed{meta['seed']}__layer{meta['layer']}__"
                    f"eps{meta['eps']}__{meta['cc']}__{meta['mode']}__k{k}.png"
                )
                fig_path = outdir / fig_rel
                label_text = label_map.get((dataset, meta["cc"], meta["tag"], meta["mode"]), {}).get(k, "")
                print("DEBUG:", dataset, meta["cc"], meta["tag"], meta["mode"], k, repr(label_text))
                save_hist(
                    cos_ds,
                    title=(
                        f"{ds} | scope={dataset} | seed={meta['seed']} layer={meta['layer']} eps={meta['eps']}\n"
                        f"{perturbation_type} | {meta['mode']} | comp={k}"
                        + (f"\nlabel: {label_text}" if label_text else "")
                    ),
                    out_path=fig_path,
                    full_range=args.full_range,
                )

                rows.append(
                    {
                        "dataset": ds,
                        "scope_dataset": dataset,
                        "global_pca": is_global,
                        "seed": int(meta["seed"]),
                        "layer": int(meta["layer"]),
                        "eps": float(meta["eps"]),
                        "cc": meta["cc"],
                        "tag": meta["tag"],
                        "mode": meta["mode"],
                        "k_selected": int(k),
                        "selection_reason": reason,
                        "top_components_considered": top_considered,
                        "corr_score_vs_delta_margin": float(corr_k_ds) if np.isfinite(corr_k_ds) else np.nan,
                        "n_vectors": int(len(cos_ds)),
                        "cos_mean": float(np.mean(cos_ds)),
                        "cos_median": float(np.median(cos_ds)),
                        "cos_p25": float(np.quantile(cos_ds, 0.25)),
                        "cos_p75": float(np.quantile(cos_ds, 0.75)),
                        "pca_path": str(pca_path),
                        "fig_path": str(fig_path),
                        "perturbation_type": perturbation_type,
                    }
                )

    if not rows:
        raise SystemExit("No matching PCA runs after filtering.")

    summary = pd.DataFrame(rows).sort_values(
        ["dataset", "perturbation_type", "k_selected", "seed", "layer", "eps", "mode"]
    )
    summary_path = outdir / "summary_component_alignment_topN_delta_margin.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
