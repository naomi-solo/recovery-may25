import os
import torch


def pca_torch(X: torch.Tensor, k: int = 10):
    """
    PCA via SVD on centered data.
    X: (N, D) float tensor on CPU
    Returns:
      components: (k, D)
      scores: (N, k)
      explained_var_ratio: (k,)
      mean: (D,)
    """
    X = X.float()
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    components = Vh[:k, :]          # (k, D)
    scores = Xc @ components.T      # (N, k)

    eigvals = (S**2) / max(1, (Xc.shape[0] - 1))
    total = eigvals.sum()
    explained = eigvals[:k] / (total + 1e-12)

    return components, scores, explained, mean.squeeze(0)


def _preview_record(rec: dict) -> str:
    ds = rec.get("dataset", "unknown")
    p = (rec.get("prompt", "") or "").replace("\n", " ").strip()
    c = (rec.get("completion", "") or "").replace("\n", " ").strip()
    ct = rec.get("completion_type", "")
    r0 = rec.get("reward_unperturbed", None)
    r1 = rec.get("reward_perturbed", None)

    head = (
        f"[{ds} | {ct}] clean={r0:.3f} adv={r1:.3f}"
        if isinstance(r0, float) and isinstance(r1, float)
        else f"[{ds} | {ct}]"
    )
    s = f"{head} | {p} || {c}"
    return s[:140]


def _select_records(records: list, mode: str) -> list:
    if mode not in ("chosen", "rejected", "both"):
        raise ValueError(f"MODE must be chosen|rejected|both, got {mode}")

    if mode == "both":
        use = records
    else:
        use = [r for r in records if r.get("completion_type") == mode]

    if len(use) == 0:
        raise RuntimeError(f"No records selected for MODE={mode}. Check your input file.")
    return use


def _load_direction_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu")
    if "records" not in obj:
        raise KeyError(f"Expected 'records' in {path}. Did you run the new extractor?")
    return obj


BASE = "results/outputs/global_verify_components"
os.makedirs(f"{BASE}/pca", exist_ok=True)

def main():
    cc_env = os.getenv("CC", None)      # "ambig" | "disambig"
    tag_env = os.getenv("TAG", None)    # e.g. "shared_flip", "shared_noflip", "separate"

    seed_default = int(os.getenv("SEED", "0"))

    # Settings (match extract_directions.py)
    layer_default = int(os.getenv("LAYER", "14"))
    eps_default = float(os.getenv("EPS", "8.0"))
    n_default = int(os.getenv("N", "200"))
    k_default = int(os.getenv("K", "10"))
    mode = os.getenv("MODE", "chosen")  # chosen|rejected|both

    cc = cc_env if cc_env is not None else "ambig"
    tag = tag_env if tag_env is not None else "flip"

    split = os.getenv("SPLIT", "train")
    dataset = os.getenv("DATASET", "bbq")

    # NEW FLAGS
    global_pca = bool(int(os.getenv("GLOBAL_PCA", "0")))
    datasets = [d.strip() for d in os.getenv("DATASETS", "bbq,gsm_mc,math_mc,mmlu,sgxs").split(",") if d.strip()]

    records_all = []
    input_paths = []
    source_objs = []

    if global_pca:
        print(f"GLOBAL_PCA=1: pooling datasets: {datasets}")
        for ds in datasets:
            in_path = (
                f"{BASE}/directions/"
                f"direction_records_{ds}_{split}_seed{seed_default}_layer{layer_default}_"
                f"eps{eps_default}_{cc}_{tag}_n{n_default}.pt"
            )
            if not os.path.exists(in_path):
                print("missing (skipping):", in_path)
                continue

            obj = _load_direction_file(in_path)
            ds_records = obj["records"]
            records_all.extend(ds_records)
            input_paths.append(in_path)
            source_objs.append(obj)

        if len(records_all) == 0:
            print("No direction files found for global PCA.")
            return

        # use metadata from first loaded file; assume shared config across pooled files
        ref_obj = source_objs[0]
        layer = int(ref_obj.get("layer", layer_default))
        eps = float(ref_obj.get("epsilon", eps_default))

        scope_name = "global"
    else:
        in_path = (
            f"{BASE}/directions/"
            f"direction_records_{dataset}_{split}_seed{seed_default}_layer{layer_default}_"
            f"eps{eps_default}_{cc}_{tag}_n{n_default}.pt"
        )
        if not os.path.exists(in_path):
            print("missing (skipping):", in_path)
            return

        obj = _load_direction_file(in_path)
        records_all = obj["records"]
        input_paths = [in_path]

        layer = int(obj.get("layer", layer_default))
        eps = float(obj.get("epsilon", eps_default))

        scope_name = dataset

    records = _select_records(records_all, mode=mode)

    X = torch.stack([r["perturbation_direction"] for r in records], dim=0)  # (N, D)
    ids = [int(r.get("pair_id", i)) for i, r in enumerate(records)]

    k_eff = min(k_default, X.shape[0], X.shape[1])
    comps, scores, evr, mean = pca_torch(X, k=k_eff)

    dataset_counts = {}
    for r in records:
        ds = r.get("dataset", "unknown")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

    out_path = (
        f"{BASE}/pca/"
        f"pca_{scope_name}_{split}_seed{seed_default}_layer{layer}_"
        f"eps{eps}_k{k_eff}_{cc}_{tag}_{mode}_n{X.shape[0]}.pt"
    )

    torch.save(
        {
            "components": comps.half(),
            "scores": scores.float(),
            "explained_var_ratio": evr.float(),
            "mean": mean.float(),
            "records": records,
            "ids": ids,
            "layer": layer,
            "epsilon": eps,
            "in_path": input_paths if global_pca else input_paths[0],
            "context_condition": cc,
            "tag": tag,
            "mode": mode,
            "requested_k": int(k_default),
            "k": int(k_eff),
            "global_pca": bool(global_pca),
            "datasets_used": sorted(dataset_counts.keys()),
            "dataset_counts": dataset_counts,
        },
        out_path,
    )

    if global_pca:
        print("loaded pooled inputs:")
        for p in input_paths:
            print("  ", p)
    else:
        print("loaded:", input_paths[0])

    print("saved:", out_path)
    print("dataset_counts:", dataset_counts)
    print("explained_var_ratio:", evr.tolist())

    # quick peek: top examples for component 0
    topk = 8
    k0 = 0
    vals, idx = torch.topk(scores[:, k0], k=min(topk, scores.shape[0]))
    print(f"\n[{scope_name} | {cc} | {tag} | mode={mode}] Top {len(idx)} examples for component {k0}:")
    for rank, (v, j) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
        print(f"{rank:02d} score={v:+.4f} id={ids[j]}  {_preview_record(records[j])}")
    print()


if __name__ == "__main__":
    main()