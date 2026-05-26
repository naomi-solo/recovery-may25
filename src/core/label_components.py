import os
import json
import torch
import glob

BASE = "results/outputs/global_verify_components"
LABELS_DIR = os.getenv("LABELS_DIR", f"{BASE}/labels")
os.makedirs(LABELS_DIR, exist_ok=True)

def render_example(rec):
    """
    Renders one PCA example for the labeling prompt.

    Supports:
      - record dict format from extract_directions.py / pca_directions.py:
          {prompt, completion, completion_type, reward_unperturbed, reward_perturbed, ...}
    """
    if isinstance(rec, str):
        return f"PROMPT: {rec.replace(chr(10), ' ')[:400]}"

    def clip(s, n):
        s = (s or "").replace("\n", " ").strip()
        return s[:n]

    dataset = rec.get("dataset", "unknown")
    prompt = clip(rec.get("prompt", ""), 320)
    completion = clip(rec.get("completion", ""), 320)
    ctype = rec.get("completion_type", "unknown")

    r0 = rec.get("reward_unperturbed", None)
    r1 = rec.get("reward_perturbed", None)

    if isinstance(r0, (float, int)) and isinstance(r1, (float, int)):
        r0f = float(r0)
        r1f = float(r1)
        dr = r1f - r0f
        reward_line = f"REWARD: clean={r0f:+.4f} adv={r1f:+.4f} Δ={dr:+.4f}"
    else:
        reward_line = "REWARD: (missing)"

    return (
        f"DATASET: {dataset}\n"
        f"TYPE: {ctype}\n"
        f"{reward_line}\n"
        f"PROMPT: {prompt}\n"
        f"COMPLETION: {completion}"
    )


def build_label_prompt(comp_id: int, top_recs, bottom_recs):
    def fmt(items):
        blocks = []
        for i, rec in enumerate(items):
            blocks.append(f"{i+1}.\n{render_example(rec)}")
        return "\n\n".join(blocks)

    return f"""You are interpreting a latent direction (a PCA component) derived from perturbation-induced representation shifts in a reward model.

Here are examples at the POSITIVE end of this direction:
{fmt(top_recs)}

Here are examples at the NEGATIVE end of this direction:
{fmt(bottom_recs)}

Task:
1) Give a short label (2–6 words).
2) Give a 1–2 sentence explanation of what this direction represents.
3) Give 8–12 keywords/phrases strongly associated with it.
4) Give 3 "NOT about" negatives (common confusions).

Be specific and avoid vague answers like "general semantics".
Do NOT describe the experiment or the model.
Only describe the semantic topic shared by the examples.
If no clear topic exists, respond "unclear".
Return JSON with keys: label, explanation, keywords, negatives.
"""


def call_openai(prompt: str, model: str = "gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text


def label_one(
    dataset: str,
    split: str,
    cc: str,
    tag: str,
    mode: str,
    layer_default: int = 14,
    eps_default: float = 8.0,
    k_default: int = 10,
    n_default: int = 200,
    model: str = "gpt-4o-mini",
):
    seed_default = int(os.getenv("SEED", "0"))

    global_pca = bool(int(os.getenv("GLOBAL_PCA", "0")))
    scope_name = "global" if global_pca else dataset

    if global_pca:
        pattern = (
            f"{BASE}/pca/"
            f"pca_{scope_name}_{split}_seed{seed_default}_layer{layer_default}_"
            f"eps{eps_default}_k{k_default}_{cc}_{tag}_{mode}_n*.pt"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            print("missing (skipping):", pattern)
            return
        pca_path = matches[-1]
    else:
        pca_path = (
            f"{BASE}/pca/"
            f"pca_{scope_name}_{split}_seed{seed_default}_layer{layer_default}_"
            f"eps{eps_default}_k{k_default}_{cc}_{tag}_{mode}_n{n_default}.pt"
        )
        if not os.path.exists(pca_path):
            print("missing (skipping):", pca_path)
            return
        
    obj = torch.load(pca_path, map_location="cpu")

    scores = obj["scores"]
    records = obj["records"]
    ids = obj["ids"]
    layer = obj["layer"]
    eps = obj["epsilon"]
    mode_used = obj.get("mode", mode)

    K = scores.shape[1]
    topM = 12
    botM = 6

    out_jsonl = os.path.join(
        LABELS_DIR,
        f"component_labels_{scope_name}_{split}_seed{seed_default}_layer{layer}_"
        f"eps{eps}_K{K}_{cc}_{tag}_{mode_used}_n{len(records)}.jsonl"
    )

    use_api = bool(os.getenv("OPENAI_API_KEY"))

    with open(out_jsonl, "w") as f:
        for k in range(K):
            col = scores[:, k]

            top_vals, top_idx = torch.topk(col, k=min(topM, col.shape[0]))
            bot_vals, bot_idx = torch.topk(-col, k=min(botM, col.shape[0]))

            top_recs = [records[i] for i in top_idx.tolist()]
            bot_recs = [records[i] for i in bot_idx.tolist()]

            prompt = build_label_prompt(k, top_recs, bot_recs)

            if use_api:
                raw = call_openai(prompt, model=model)
                clean = raw.replace("```json", "").replace("```", "").strip()

                try:
                    data = json.loads(clean)
                except Exception:
                    data = {
                        "label": "PARSE_ERROR",
                        "explanation": raw,
                        "keywords": [],
                        "negatives": [],
                    }
            else:
                print("\n" + "=" * 80)
                print(f"[{scope_name} | {cc} | {tag} | mode={mode_used}] COMPONENT {k} PROMPT:\n")
                print(prompt)
                data = {"label": "MANUAL", "explanation": "", "keywords": [], "negatives": []}

            row = {
                "component": int(k),
                "layer": int(layer),
                "epsilon": float(eps),
                "context_condition": cc,
                "tag": tag,
                "mode": mode_used,
                "global_pca": bool(global_pca),
                "pca_path": pca_path,
                "datasets_used": obj.get("datasets_used", []),
                "dataset_counts": obj.get("dataset_counts", {}),
                "top_ids": [ids[i] for i in top_idx.tolist()],
                "top_scores": [float(v) for v in top_vals.tolist()],
                "bottom_ids": [ids[i] for i in bot_idx.tolist()],
                "bottom_scores": [float(v) for v in col[bot_idx].tolist()],
                **data,
            }
            f.write(json.dumps(row) + "\n")

    print("wrote:", out_jsonl)


def main():
    dataset = os.getenv("DATASET", "bbq")
    split = os.getenv("SPLIT", "train")
    cc_env = os.getenv("CC", None)
    tag_env = os.getenv("TAG", None)

    layer_default = int(os.getenv("LAYER", "14"))
    eps_default = float(os.getenv("EPS", "8.0"))
    n_default = int(os.getenv("N", "200"))
    k_default = int(os.getenv("K", "10"))

    mode = os.getenv("MODE", "chosen")

    cc = cc_env if cc_env is not None else "ambig"
    tag = tag_env if tag_env is not None else "flip"

    label_one(
        dataset=dataset,
        split=split,
        cc=cc,
        tag=tag,
        mode=mode,
        layer_default=layer_default,
        eps_default=eps_default,
        k_default=k_default,
        n_default=n_default if mode != "both" else 2 * n_default,
    )


if __name__ == "__main__":
    main()