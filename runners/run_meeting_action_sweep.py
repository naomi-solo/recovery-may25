#!/usr/bin/env python3
"""
Run a meeting-focused perturbation sweep.

This is a deadline-friendly wrapper around the existing extraction and PCA
pipeline. It runs all direction extraction first, then global PCA, so GLOBAL_PCA
actually pools all requested datasets instead of depending on loop order.

Examples:
  python runners/run_meeting_action_sweep.py --quick
  python runners/run_meeting_action_sweep.py --n-total 40 --skip-labels
  python runners/run_meeting_action_sweep.py --n-total 200 --with-labels
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/outputs/global_verify_components"

DEFAULT_DATASETS = ["bbq", "math_mc", "mmlu"]
FULL_DATASETS = ["bbq", "gsm_mc", "math_mc", "mmlu", "sgxs"]
DEFAULT_EPS = [0.1, 1.0]
DEFAULT_REGIMES = ["shared_noflip", "shared_flip"]
ALL_REGIMES = ["shared_noflip", "shared_flip", "separate"]

REGIME_CONFIG = {
    "shared_noflip": {"perturbation_mode": "shared", "sign_flip": 0},
    "shared_flip": {"perturbation_mode": "shared", "sign_flip": 1},
    "separate": {"perturbation_mode": "separate", "sign_flip": 0},
}

MODES = ["chosen", "rejected"]
FIXED_ENV = {
    "PGD_STEPS": "8",
    "PER_TOKEN": "1",
    "STEP_SIZE": "1.0",
    "MAX_LENGTH": "2048",
    "PYTHONPATH": str(ROOT),
}


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def run_script(script: str, env: dict[str, object]) -> None:
    full_env = os.environ.copy()
    full_env.update(FIXED_ENV)
    full_env.update({k: str(v) for k, v in env.items()})

    shown = {k: full_env[k] for k in sorted(env)}
    print("\n" + "=" * 88)
    print(f"RUN {script}")
    print(shown)

    t0 = time.time()
    proc = subprocess.run(["python", script], cwd=ROOT, env=full_env)
    minutes = (time.time() - t0) / 60.0
    print(f"DONE rc={proc.returncode} minutes={minutes:.1f}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def direction_path(dataset: str, split: str, seed: int, layer: int, eps: float, cc: str, tag: str, n_total: int) -> Path:
    return BASE / "directions" / (
        f"direction_records_{dataset}_{split}_seed{seed}_layer{layer}_"
        f"eps{eps}_{cc}_{tag}_n{n_total}.pt"
    )


def pca_glob(split: str, seed: int, layer: int, eps: float, k: int, cc: str, tag: str, mode: str) -> str:
    return str(
        BASE / "pca" / (
            f"pca_global_{split}_seed{seed}_layer{layer}_eps{eps}_"
            f"k{k}_{cc}_{tag}_{mode}_n*.pt"
        )
    )


def label_glob(split: str, seed: int, layer: int, eps: float, k: int, cc: str, tag: str, mode: str) -> str:
    return str(
        BASE / "labels" / (
            f"component_labels_global_{split}_seed{seed}_layer{layer}_eps{eps}_"
            f"K{k}_{cc}_{tag}_{mode}_n*.jsonl"
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Use N=25, three datasets, two regimes; good for a meeting pilot.")
    ap.add_argument("--full", action="store_true", help="Use all five datasets and all three regimes.")
    ap.add_argument("--n-total", type=int, default=None)
    ap.add_argument("--datasets", default=None, help="Comma list. Default quick set: bbq,math_mc,mmlu")
    ap.add_argument("--eps", default="0.1,1.0", help="Comma list of epsilon values.")
    ap.add_argument("--regimes", default=None, help="Comma list from shared_noflip,shared_flip,separate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=20, help="Number of PCA components to keep.")
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cc", default="ambig")
    ap.add_argument("--with-labels", action="store_true", help="Call OpenAI labeler after PCA. Requires OPENAI_API_KEY.")
    ap.add_argument("--skip-labels", action="store_true", help="Explicitly skip labels.")
    args = ap.parse_args()

    n_total = args.n_total if args.n_total is not None else (25 if args.quick else 40)
    datasets = parse_csv(args.datasets) if args.datasets else (FULL_DATASETS if args.full else DEFAULT_DATASETS)
    regimes = parse_csv(args.regimes) if args.regimes else (ALL_REGIMES if args.full else DEFAULT_REGIMES)
    eps_values = parse_float_csv(args.eps)

    unknown = [r for r in regimes if r not in REGIME_CONFIG]
    if unknown:
        raise SystemExit(f"Unknown regimes: {unknown}. Choose from {sorted(REGIME_CONFIG)}")

    run_labels = args.with_labels and not args.skip_labels
    if run_labels and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--with-labels requires OPENAI_API_KEY")

    print("Meeting sweep config")
    print({
        "datasets": datasets,
        "eps": eps_values,
        "regimes": regimes,
        "n_total": n_total,
        "k": args.k,
        "labels": run_labels,
    })

    for eps in eps_values:
        for tag in regimes:
            regime = REGIME_CONFIG[tag]

            for dataset in datasets:
                out = direction_path(dataset, args.split, args.seed, args.layer, eps, args.cc, tag, n_total)
                if out.exists():
                    print(f"SKIP direction exists: {out}")
                    continue

                run_script(
                    "src/core/extract_directions.py",
                    {
                        "DATASET": dataset,
                        "SPLIT": args.split,
                        "HOLDOUT": args.holdout,
                        "SEED": args.seed,
                        "EPS": eps,
                        "LAYER": args.layer,
                        "CC": args.cc,
                        "TAG": tag,
                        "SIGN_FLIP": regime["sign_flip"],
                        "PERTURBATION_MODE": regime["perturbation_mode"],
                        "N_TOTAL": n_total,
                    },
                )

            for mode in MODES:
                existing_pca = glob.glob(pca_glob(args.split, args.seed, args.layer, eps, args.k, args.cc, tag, mode))
                if existing_pca:
                    print(f"SKIP global PCA exists: {existing_pca[-1]}")
                else:
                    run_script(
                        "src/core/pca_directions.py",
                        {
                            "GLOBAL_PCA": 1,
                            "DATASETS": ",".join(datasets),
                            "DATASET": "global",
                            "SPLIT": args.split,
                            "SEED": args.seed,
                            "EPS": eps,
                            "LAYER": args.layer,
                            "CC": args.cc,
                            "TAG": tag,
                            "MODE": mode,
                            "K": args.k,
                            "N": n_total,
                        },
                    )

                if not run_labels:
                    continue

                existing_labels = glob.glob(label_glob(args.split, args.seed, args.layer, eps, args.k, args.cc, tag, mode))
                if existing_labels:
                    print(f"SKIP labels exist: {existing_labels[-1]}")
                else:
                    run_script(
                        "src/core/label_components.py",
                        {
                            "GLOBAL_PCA": 1,
                            "DATASET": "global",
                            "SPLIT": args.split,
                            "SEED": args.seed,
                            "EPS": eps,
                            "LAYER": args.layer,
                            "CC": args.cc,
                            "TAG": tag,
                            "MODE": mode,
                            "K": args.k,
                            "N": n_total,
                        },
                    )

    print("\nMeeting sweep complete.")
    print("Next: python analysis/make_meeting_alignment_summary.py --top-n 20")


if __name__ == "__main__":
    main()
