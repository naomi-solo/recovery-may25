#run_perturb_sweep.py

import os
import itertools
import subprocess
import time
import json
from datetime import datetime

# -------------------------
# CORE SWEEP CONFIG
# -------------------------
SEEDS = [42]
EPS_LIST = [0.5]
LAYERS = [14]  # mid layer for Skywork-Reward-V2-Qwen3-0.6B
N_TOTAL = 200
HOLDOUT_SIZE = 50
SPLIT = "train"

DATASETS = ["bbq", "gsm_mc", "math_mc", "mmlu", "sgxs"]
REGIMES = [
    {"name": "shared_noflip", "perturbation_mode": "shared", "sign_flip": 0, "tag": "shared_noflip"},
    {"name": "shared_flip", "perturbation_mode": "shared", "sign_flip": 1, "tag": "shared_flip"},
    {"name": "separate", "perturbation_mode": "separate", "sign_flip": 0, "tag": "separate"},
]

FIXED = {
    "PGD_STEPS": "8",
    "PER_TOKEN": "1",
    "STEP_SIZE": "1.0",
    "MAX_LENGTH": "2048",
}

K_DEFAULT = "10"
MODES = ["chosen", "rejected"]

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = str(ROOT / "src/core/extract_directions.py")
PCA_SCRIPT = str(ROOT / "src/core/pca_directions.py")
LABEL_SCRIPT = str(ROOT / "src/core/label_components.py")

LOG_DIR = "results/data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# IMPORTANT: use a stable log name so resumes append to the same file
LOG_PATH = os.path.join(LOG_DIR, "pipeline_resume.jsonl")

TAIL_CHARS = 2000


def run_script(script: str, env_overrides: dict) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}:{old_pp}" if old_pp else str(ROOT)

    t0 = time.time()
    try:
        p = subprocess.run(
            ["python", script],
            env=env,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        dt = time.time() - t0
        return {
            "script": script,
            "ok": (p.returncode == 0),
            "return_code": p.returncode,
            "seconds": round(dt, 3),
            "stdout_tail": (p.stdout or "")[-TAIL_CHARS:],
            "stderr_tail": (p.stderr or "")[-TAIL_CHARS:],
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "script": script,
            "ok": False,
            "return_code": -1,
            "seconds": round(dt, 3),
            "stdout_tail": "",
            "stderr_tail": repr(e)[-TAIL_CHARS:],
        }


def log_row(row: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def paths_for_config(dataset, split, seed, eps, layer, cc, tag, n_total, k_default):
    BASE = "results/outputs/global_verify_components"

    directions_path = (
        f"{BASE}/directions/"
        f"direction_records_{dataset}_{split}_seed{seed}_layer{layer}_eps{eps}_{cc}_{tag}_n{n_total}.pt"
    )

    pca_paths = {}
    label_paths = {}
    for mode in MODES:
        pca_paths[mode] = (
            f"{BASE}/pca/"
            f"pca_{dataset}_{split}_seed{seed}_layer{layer}_eps{eps}_"
            f"k{k_default}_{cc}_{tag}_{mode}_n{n_total}.pt"
        )
        label_paths[mode] = (
            f"{BASE}/labels/"
            f"component_labels_{dataset}_{split}_seed{seed}_layer{layer}_"
            f"eps{eps}_K{k_default}_{cc}_{tag}_{mode}_n{n_total}.jsonl"
        )

    return directions_path, pca_paths, label_paths


def config_is_done(dataset, split, seed, eps, layer, cc, tag):
    directions_path, pca_paths, label_paths = paths_for_config(
        dataset, split, seed, eps, layer, cc, tag, N_TOTAL, K_DEFAULT
    )

    if not os.path.exists(directions_path):
        return False

    for mode in MODES:
        if not os.path.exists(pca_paths[mode]):
            return False
        if not os.path.exists(label_paths[mode]):
            return False

    return True


def main():
    # Match run_eps_sweep-style setup:
    # - dataset loop
    # - fixed eps=0.5
    # - three perturbation regimes
    # We keep a single context value for compatibility with existing record schema/filenames.
    contexts = ["ambig"]
    grid = list(itertools.product(DATASETS, SEEDS, EPS_LIST, LAYERS, contexts, REGIMES))
    total = len(grid)

    print("Running sweep with resume-by-files.")
    print("Total configs:", total)
    print("Modes:", MODES)
    print("Log file:", LOG_PATH)

    failures = 0
    skipped = 0

    for i, (dataset, seed, eps, layer, cc, regime) in enumerate(grid, 1):
        sign_flip = regime["sign_flip"]
        tag = regime["tag"]
        perturbation_mode = regime["perturbation_mode"]

        if config_is_done(dataset, SPLIT, seed, eps, layer, cc, tag):
            skipped += 1
            print(
                f"[{i}/{total}] SKIP already done: "
                f"DATASET={dataset} SPLIT={SPLIT} SEED={seed} EPS={eps} LAYER={layer} "
                f"CC={cc} TAG={tag} MODE={perturbation_mode} SIGN_FLIP={sign_flip}"
            )
            continue

        cfg_env = {
            "DATASET": dataset,
            "SPLIT": SPLIT,
            "HOLDOUT": HOLDOUT_SIZE,
            "SEED": seed,
            "EPS": eps,
            "LAYER": layer,
            "CC": cc,
            "TAG": tag,
            "SIGN_FLIP": sign_flip,
            "PERTURBATION_MODE": perturbation_mode,
            "N_TOTAL": N_TOTAL,
            "GLOBAL_PCA": "1",
            "DATASETS": ",".join(DATASETS),
            **FIXED,
            "K": K_DEFAULT,
            "N": str(N_TOTAL),
        }

        print("\n" + "=" * 80)
        print(
            f"[{i}/{total}] RUN: "
            f"DATASET={dataset} SPLIT={SPLIT} SEED={seed} EPS={eps} LAYER={layer} "
            f"CC={cc} TAG={tag} MODE={perturbation_mode} SIGN_FLIP={sign_flip}"
        )

        row = {
            "ts_utc": datetime.utcnow().isoformat(),
            "config": {
                "seed": seed,
                "dataset": dataset,
                "split": SPLIT,
                "eps": eps,
                "layer": layer,
                "cc": cc,
                "tag": tag,
                "sign_flip": sign_flip,
                "perturbation_mode": perturbation_mode,
                "n_total": N_TOTAL,
                "k": K_DEFAULT,
            },
            "steps": {"extract": None, "pca": {}, "label": {}},
        }

        # extract
        res_extract = run_script(EXTRACT_SCRIPT, cfg_env)
        row["steps"]["extract"] = res_extract
        if not res_extract["ok"]:
            failures += 1
            print(f"✗ extract FAILED rc={res_extract['return_code']}")
            log_row(row)
            continue
        print(f"✓ extract ok ({res_extract['seconds']/60:.2f} min)")

        # pca + label for both modes
        for mode in MODES:
            cfg_env["MODE"] = mode

            res_pca = run_script(PCA_SCRIPT, cfg_env)
            row["steps"]["pca"][mode] = res_pca
            if not res_pca["ok"]:
                failures += 1
                print(f"✗ pca ({mode}) FAILED rc={res_pca['return_code']}")
                row["steps"]["label"][mode] = {"skipped": True, "reason": "pca failed"}
                continue
            print(f"✓ pca ({mode}) ok ({res_pca['seconds']/60:.2f} min)")

            res_lab = run_script(LABEL_SCRIPT, cfg_env)
            row["steps"]["label"][mode] = res_lab
            if not res_lab["ok"]:
                failures += 1
                print(f"✗ label ({mode}) FAILED rc={res_lab['return_code']}")
            else:
                print(f"✓ label ({mode}) ok ({res_lab['seconds']/60:.2f} min)")

        log_row(row)

    print("\nDone.")
    print("Skipped (already done):", skipped)
    print("Failures:", failures)
    print("Log saved to:", LOG_PATH)


if __name__ == "__main__":
    main()
