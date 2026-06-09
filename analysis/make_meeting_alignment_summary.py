#!/usr/bin/env python3
"""
Create meeting-ready alignment summaries from PCA artifacts.

This script first runs verify_component_alignment_v4.py, then reshapes the
resulting summary into compact CSVs and a short markdown takeaway file.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_alignment(args: argparse.Namespace) -> Path:
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary_component_alignment_topN_delta_margin.csv"

    if args.skip_verify and summary_path.exists():
        return summary_path

    cmd = [
        "python",
        "analysis/verify_component_alignment_v4.py",
        "--pca_glob",
        args.pca_glob,
        "--labels_dir",
        args.labels_dir,
        "--top_n",
        str(args.top_n),
        "--outdir",
        args.outdir,
    ]
    if args.full_range:
        cmd.append("--full_range")

    subprocess.run(cmd, cwd=ROOT, check=True)
    return summary_path


def fmt_float(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def write_takeaways(df: pd.DataFrame, outdir: Path) -> None:
    lines = ["# Meeting Alignment Takeaways", ""]

    if df.empty:
        lines.append("No rows were available in the alignment summary.")
        (outdir / "meeting_takeaways.md").write_text("\n".join(lines) + "\n")
        return

    lines.append("## Strongest Median Alignments")
    top = df.sort_values("cos_median", ascending=False).head(12)
    for _, r in top.iterrows():
        label = r.get("selected_component_label", "") or "unlabeled"
        lines.append(
            f"- {r['dataset']} | eps={r['eps']} | {r['tag']} | {r['mode']}: "
            f"median={fmt_float(r['cos_median'])}, k={int(r['k_selected'])}, label={label}"
        )

    lines.append("")
    lines.append("## Shared No-Flip Check")
    rank_cols = ["dataset", "eps", "mode", "tag", "cos_median"]
    ranked = df[rank_cols].copy()
    ranked["rank_within_dataset_eps_mode"] = ranked.groupby(["dataset", "eps", "mode"])["cos_median"].rank(
        method="min", ascending=False
    )
    snf = ranked[ranked["tag"] == "shared_noflip"].sort_values(["dataset", "eps", "mode"])
    if snf.empty:
        lines.append("- No shared_noflip rows found.")
    else:
        for _, r in snf.iterrows():
            lines.append(
                f"- {r['dataset']} | eps={r['eps']} | {r['mode']}: "
                f"rank {int(r['rank_within_dataset_eps_mode'])}, median={fmt_float(r['cos_median'])}"
            )

    if df["eps"].nunique() > 1:
        lines.append("")
        lines.append("## Epsilon Comparison")
        pivot = df.pivot_table(
            index=["dataset", "tag", "mode"], columns="eps", values="cos_median", aggfunc="mean"
        )
        eps_cols = sorted([c for c in pivot.columns if isinstance(c, float)])
        if len(eps_cols) >= 2:
            lo, hi = eps_cols[0], eps_cols[-1]
            pivot["delta_hi_minus_lo"] = pivot[hi] - pivot[lo]
            for idx, row in pivot.sort_values("delta_hi_minus_lo", ascending=False).head(10).iterrows():
                dataset, tag, mode = idx
                lines.append(
                    f"- {dataset} | {tag} | {mode}: eps {lo}={fmt_float(row[lo])}, "
                    f"eps {hi}={fmt_float(row[hi])}, delta={fmt_float(row['delta_hi_minus_lo'])}"
                )

    (outdir / "meeting_takeaways.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pca_glob",
        default="results/outputs/global_verify_components/pca/pca*.pt",
        help="PCA glob passed to verify_component_alignment_v4.py",
    )
    ap.add_argument(
        "--labels_dir",
        default="results/outputs/global_verify_components/labels",
    )
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--outdir", default="results/analysis/meeting_alignment_summary")
    ap.add_argument("--skip-verify", action="store_true", help="Use an existing summary CSV in outdir.")
    ap.add_argument("--full-range", action="store_true")
    args = ap.parse_args()

    summary_path = run_alignment(args)
    outdir = ROOT / args.outdir
    df = pd.read_csv(summary_path)

    keep = [
        "dataset",
        "eps",
        "tag",
        "mode",
        "k_selected",
        "selected_component_label",
        "corr_score_vs_delta_margin",
        "n_vectors",
        "cos_mean",
        "cos_median",
        "cos_p25",
        "cos_p75",
    ]
    keep = [c for c in keep if c in df.columns]
    long_df = df[keep].sort_values(["dataset", "eps", "tag", "mode"])
    long_path = outdir / "meeting_medians_long.csv"
    long_df.to_csv(long_path, index=False)

    pivot_path = outdir / "meeting_medians_pivot.csv"
    pivot = df.pivot_table(
        index=["dataset", "eps", "mode"], columns="tag", values="cos_median", aggfunc="mean"
    ).reset_index()
    pivot.to_csv(pivot_path, index=False)

    rank_path = outdir / "shared_noflip_rank.csv"
    ranks = df[["dataset", "eps", "mode", "tag", "cos_median"]].copy()
    ranks["rank_within_dataset_eps_mode"] = ranks.groupby(["dataset", "eps", "mode"])["cos_median"].rank(
        method="min", ascending=False
    )
    ranks = ranks.sort_values(["dataset", "eps", "mode", "rank_within_dataset_eps_mode"])
    ranks.to_csv(rank_path, index=False)

    eps_path = outdir / "epsilon_delta_by_dataset_tag_mode.csv"
    eps_pivot = df.pivot_table(
        index=["dataset", "tag", "mode"], columns="eps", values="cos_median", aggfunc="mean"
    )
    eps_cols = sorted([c for c in eps_pivot.columns if isinstance(c, float)])
    if len(eps_cols) >= 2:
        eps_pivot["delta_max_minus_min_eps"] = eps_pivot[eps_cols[-1]] - eps_pivot[eps_cols[0]]
    eps_pivot.reset_index().to_csv(eps_path, index=False)

    write_takeaways(df, outdir)

    print("Wrote:")
    for p in [summary_path, long_path, pivot_path, rank_path, eps_path, outdir / "meeting_takeaways.md"]:
        print(" ", p)

    print("\nPreview:")
    preview_cols = ["dataset", "eps", "tag", "mode", "k_selected", "selected_component_label", "cos_median"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].sort_values("cos_median", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
