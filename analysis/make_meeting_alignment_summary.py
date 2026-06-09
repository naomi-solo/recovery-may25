#!/usr/bin/env python3
"""Make meeting-ready CSVs and a short markdown summary from alignment outputs."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_verify(args: argparse.Namespace) -> Path:
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "summary_component_alignment_topN_delta_margin.csv"
    if args.skip_verify and summary.exists():
        return summary
    subprocess.run([
        "python", "analysis/verify_component_alignment_v4.py",
        "--pca_glob", args.pca_glob,
        "--labels_dir", args.labels_dir,
        "--top_n", str(args.top_n),
        "--outdir", args.outdir,
    ], cwd=ROOT, check=True)
    return summary


def f3(x) -> str:
    return "NA" if pd.isna(x) else f"{float(x):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_glob", default="results/outputs/global_verify_components/pca/pca*.pt")
    ap.add_argument("--labels_dir", default="results/outputs/global_verify_components/labels")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--outdir", default="results/analysis/meeting_alignment_summary")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    summary_path = run_verify(args)
    df = pd.read_csv(summary_path)

    keep = [
        "dataset", "eps", "tag", "mode", "k_selected", "selected_component_label",
        "corr_score_vs_delta_margin", "n_vectors", "cos_mean", "cos_median", "cos_p25", "cos_p75",
    ]
    keep = [c for c in keep if c in df.columns]
    long_df = df[keep].sort_values(["dataset", "eps", "tag", "mode"])
    long_path = outdir / "meeting_medians_long.csv"
    long_df.to_csv(long_path, index=False)

    pivot = df.pivot_table(index=["dataset", "eps", "mode"], columns="tag", values="cos_median", aggfunc="mean").reset_index()
    pivot_path = outdir / "meeting_medians_pivot.csv"
    pivot.to_csv(pivot_path, index=False)

    ranks = df[["dataset", "eps", "mode", "tag", "cos_median"]].copy()
    ranks["rank_within_dataset_eps_mode"] = ranks.groupby(["dataset", "eps", "mode"])["cos_median"].rank(method="min", ascending=False)
    rank_path = outdir / "shared_noflip_rank.csv"
    ranks.sort_values(["dataset", "eps", "mode", "rank_within_dataset_eps_mode"]).to_csv(rank_path, index=False)

    eps_pivot = df.pivot_table(index=["dataset", "tag", "mode"], columns="eps", values="cos_median", aggfunc="mean")
    eps_cols = sorted([c for c in eps_pivot.columns if isinstance(c, float)])
    if len(eps_cols) >= 2:
        eps_pivot["delta_max_minus_min_eps"] = eps_pivot[eps_cols[-1]] - eps_pivot[eps_cols[0]]
    eps_path = outdir / "epsilon_delta_by_dataset_tag_mode.csv"
    eps_pivot.reset_index().to_csv(eps_path, index=False)

    lines = ["# Meeting Alignment Takeaways", "", "## Strongest Median Alignments"]
    for _, r in df.sort_values("cos_median", ascending=False).head(12).iterrows():
        label = r.get("selected_component_label", "") or "unlabeled"
        lines.append(f"- {r['dataset']} | eps={r['eps']} | {r['tag']} | {r['mode']}: median={f3(r['cos_median'])}, k={int(r['k_selected'])}, label={label}")

    lines += ["", "## Shared No-Flip Ranks"]
    snf = ranks[ranks["tag"] == "shared_noflip"].sort_values(["dataset", "eps", "mode"])
    if snf.empty:
        lines.append("- No shared_noflip rows found.")
    else:
        for _, r in snf.iterrows():
            lines.append(f"- {r['dataset']} | eps={r['eps']} | {r['mode']}: rank {int(r['rank_within_dataset_eps_mode'])}, median={f3(r['cos_median'])}")

    takeaway_path = outdir / "meeting_takeaways.md"
    takeaway_path.write_text("\n".join(lines) + "\n")

    print("Wrote:")
    for p in [summary_path, long_path, pivot_path, rank_path, eps_path, takeaway_path]:
        print(" ", p)
    print("\nPreview:")
    preview = ["dataset", "eps", "tag", "mode", "k_selected", "selected_component_label", "cos_median"]
    preview = [c for c in preview if c in df.columns]
    print(df[preview].sort_values("cos_median", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
