#!/usr/bin/env python3
"""Deadline-friendly runner for the meeting action items."""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/outputs/global_verify_components"
FULL_DATASETS = ["bbq", "gsm_mc", "math_mc", "mmlu", "sgxs"]
REGIMES = {
    "shared_noflip": {"PERTURBATION_MODE": "shared", "SIGN_FLIP": 0},
    "shared_flip": {"PERTURBATION_MODE": "shared", "SIGN_FLIP": 1},
    "separate": {"PERTURBATION_MODE": "separate", "SIGN_FLIP": 0},
}
MODES = ["chosen", "rejected"]
FIXED = {
    "PGD_STEPS": "8",
    "PER_TOKEN": "1",
    "STEP_SIZE": "1.0",
    "MAX_LENGTH": "2048",
    "PYTHONPATH": str(ROOT),
}


def csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run(script: str, env: dict[str, object]) -> None:
    e = os.environ.copy()
    e.update(FIXED)
    e.update({k: str(v) for k, v in env.items()})
    print("\n" + "=" * 88)
    print("RUN", script)
    print({k: e[k] for k in sorted(env)})
    t0 = time.time()
    p = subprocess.run(["python", script], cwd=ROOT, env=e)
    print(f"DONE rc={p.returncode} minutes={(time.time() - t0) / 60:.1f}")
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def direction_path(ds: str, split: str, seed: int, layer: int, eps: float, cc: str, tag: str, n: int) -> Path:
    return BASE / "directions" / f"direction_records_{ds}_{split}_seed{seed}_layer{layer}_eps{eps}_{cc}_{tag}_n{n}.pt"


def pca_glob(split: str, seed: int, layer: int, eps: float, k: int, cc: str, tag: str, mode: str) -> str:
    return str(BASE / "pca" / f"pca_global_{split}_seed{seed}_layer{layer}_eps{eps}_k{k}_{cc}_{tag}_{mode}_n*.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Use all datasets and all perturbation styles.")
    ap.add_argument("--n-total", type=int, default=25)
    ap.add_argument("--datasets", default=",".join(FULL_DATASETS))
    ap.add_argument("--eps", default="0.1,1.0")
    ap.add_argument("--regimes", default="shared_noflip,shared_flip,separate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--holdout", type=int, default=50)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cc", default="ambig")
    ap.add_argument("--with-labels", action="store_true")
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()

    datasets = csv(args.datasets)
    regimes = csv(args.regimes)
    eps_values = floats(args.eps)
    bad = [r for r in regimes if r not in REGIMES]
    if bad:
        raise SystemExit(f"Unknown regimes {bad}; choose from {sorted(REGIMES)}")
    if args.with_labels and args.skip_labels:
        raise SystemExit("Use only one of --with-labels / --skip-labels")
    if args.with_labels and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("--with-labels requires OPENAI_API_KEY")

    print("Meeting sweep config")
    print({"datasets": datasets, "eps": eps_values, "regimes": regimes, "n_total": args.n_total, "k": args.k, "labels": args.with_labels})

    for eps in eps_values:
        for tag in regimes:
            for ds in datasets:
                out = direction_path(ds, args.split, args.seed, args.layer, eps, args.cc, tag, args.n_total)
                if out.exists():
                    print("SKIP direction exists:", out)
                else:
                    run("src/core/extract_directions.py", {
                        "DATASET": ds,
                        "SPLIT": args.split,
                        "HOLDOUT": args.holdout,
                        "SEED": args.seed,
                        "EPS": eps,
                        "LAYER": args.layer,
                        "CC": args.cc,
                        "TAG": tag,
                        "N_TOTAL": args.n_total,
                        **REGIMES[tag],
                    })

            for mode in MODES:
                matches = glob.glob(pca_glob(args.split, args.seed, args.layer, eps, args.k, args.cc, tag, mode))
                if matches:
                    print("SKIP global PCA exists:", matches[-1])
                else:
                    run("src/core/pca_directions.py", {
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
                        "N": args.n_total,
                    })

                if args.with_labels:
                    run("src/core/label_components.py", {
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
                        "N": args.n_total,
                    })

    print("\nDone. Next run: python analysis/make_meeting_alignment_summary.py --top-n 20")


if __name__ == "__main__":
    main()
