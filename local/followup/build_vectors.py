"""Direction set for the follow-up experiments (spec section "Direction set needed across experiments").

Reuses:
  - pain (S2/S1), fear, sadness, negative emotion: the paper's own 3.2/02 recipe (control_vec / compute_pain_vector,
    copied verbatim below since that script is written to run as __main__, not import), at the exp3/4 injection
    layer (the S2 ladder's layer, 25) for steering, and the existing vectors_full_steering (S1 ladder layer, 19)
    for exp1/2 readout.
  - random x10: the same Gaussian-normalized-to-the-pain-vector's-norm recipe already used in selfmed.py /
    run_selfmod_narration.py, same 10 fixed seeds as the paper's script (RAND_SEEDS).
  - vertigo, hunger: new. Extracted with extract_gguf over datasets/followup/{vertigo,hunger}_100.json, then the
    same control_vec recipe (mean(condition) - pooled-neutral mean, denoised against the neutral PCA basis) at
    layer 25, using the neutral pool (S1_1P/S2_1P/ControlSupplement_1P category D) already in the 3.2 activations.pt
    (no re-extraction needed for the neutral side).

Validates every new vector: in-sample AUC vs the same neutral pool, and cosine with pain(S2)/fear/sadness.
Writes results/followup/vectors/direction_set_L25.pt (steering set) and a validation table/plot.

    .venv-Pain-axis/bin/python local/followup/build_vectors.py
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import selfmed  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
INJ_LAYER = 25                          # the S2 ladder's layer, shared by all directions used for steering (exp3/4)
DENOISE_VARIANCE = 0.5
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
OUT = M.REPO_ROOT / "results" / "followup" / "vectors"
DS = M.REPO_ROOT / "datasets" / "followup"


def clean_mean(x):
    return np.nanmean(np.where(np.isinf(x), np.nan, x), axis=0)


def denoise_basis(neutral_acts, neutral_mean):
    X = np.nan_to_num(neutral_acts - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    return Vt[:min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))]


def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec


def compute_pain_vector(acts, cats):
    cats = np.array(cats)
    pain_mean = np.nanmean(acts[np.isin(cats, PAIN_CATEGORIES)], axis=0)
    ctrl = acts[~np.isin(cats, PAIN_CATEGORIES)]
    ctrl_mean = np.nanmean(ctrl, axis=0)
    vec = np.nan_to_num(pain_mean - ctrl_mean, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA().fit(ctrl - ctrl_mean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for d in pca.components_[:min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))]:
        vec = vec - np.dot(vec, d) * d
    return vec


def extract_new_sentences():
    """Runs extract_gguf once over vertigo + hunger prompts; returns {name: (acts[100, 64, D], categories, sets)}."""
    work = M.RUN_DIR / "followup_extract"
    work.mkdir(exist_ok=True)
    out = {}
    all_prompts, spans = [], {}
    for tag in ("vertigo", "hunger"):
        d = json.load(open(DS / f"{tag}_100.json", encoding="utf-8"))["datasets"][f"{tag.capitalize()}_1P"]["sentences"]
        spans[tag] = (len(all_prompts), len(all_prompts) + len(d), [s["category"] for s in d], [s["set"] for s in d])
        all_prompts += [s["prompt"] for s in d]
    assert all("\n" not in p for p in all_prompts)
    (work / "prompts.txt").write_text("\n".join(all_prompts) + "\n", encoding="utf-8")
    if not (work / "meta.json").exists():
        subprocess.run([str(M.RUN_DIR / "extract_gguf"), str(GGUF), str(work / "prompts.txt"), str(work), "--ts", "0.45,0.55"], check=True)
    meta = json.load(open(work / "meta.json"))
    arr = np.fromfile(work / "final.f32", dtype=np.float32).reshape(meta["n_prompts"], meta["n_layers"], meta["n_embd"])
    assert meta["n_prompts"] == len(all_prompts)
    for tag, (i0, i1, cats, sets) in spans.items():
        out[tag] = (arr[i0:i1], cats, sets)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    saved = torch.load(M.RUN_DIR / "results" / NAME / "activations.pt", mmap=True, weights_only=True)
    ft, meta = saved["activations"]["final_token"], saved["metadata"]

    def rows(ds, cats=None):
        a = ft[ds][INJ_LAYER].numpy()
        if cats is None:
            return a
        return a[np.isin(np.array(meta[ds]["categories"]), cats)]

    neutral = np.concatenate([rows(ds, ["D"]) for ds in S_SETS])
    neutral_mean = clean_mean(neutral)
    basis = denoise_basis(neutral, neutral_mean)

    def control_vec(acts):
        return project_out(np.nan_to_num(clean_mean(acts) - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0), basis)

    vectors = {
        "s2_pain_vector": torch.tensor(compute_pain_vector(rows("S2_1P"), meta["S2_1P"]["categories"])),
        "s1_pain_vector": torch.tensor(compute_pain_vector(rows("S1_1P"), meta["S1_1P"]["categories"])),
        "fear_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["B"]) for ds in S_SETS]))),
        "negemotion_vector": torch.tensor(control_vec(np.concatenate([rows(ds, ["C1"]) for ds in S_SETS]))),
        "sadness_vector": torch.tensor(control_vec(rows("SD_sadness_1P"))) if "SD_sadness_1P" in ft else None,
    }
    vectors = {k: v for k, v in vectors.items() if v is not None}

    new = extract_new_sentences()
    val_rows = []
    for tag in ("vertigo", "hunger"):
        acts, cats, sets = new[tag]
        acts_L = acts[:, INJ_LAYER]
        v = control_vec(acts_L)
        vectors[f"{tag}_vector"] = torch.tensor(v)
        unit = v / np.linalg.norm(v)
        proj_pos = acts_L @ unit
        proj_neg = neutral @ unit
        auc = roc_auc_score(np.concatenate([np.ones(len(proj_pos)), np.zeros(len(proj_neg))]), np.concatenate([proj_pos, proj_neg]))
        row = {"vector": tag, "auc_vs_pooled_neutral": round(float(auc), 4), "norm": round(float(np.linalg.norm(v)), 2)}
        for other in ("s2_pain_vector", "fear_vector", "negemotion_vector"):
            if other in vectors:
                ov = vectors[other].numpy()
                row[f"cos_{other}"] = round(float(v @ ov / (np.linalg.norm(v) * np.linalg.norm(ov))), 4)
        val_rows.append(row)

    # random x10: same recipe as selfmed.py / the paper's script, normalized to the S2 pain vector's norm, at this layer
    RAND_SEEDS = None
    for n in ast.parse((M.REPO_ROOT / "scripts" / "4.3_selfmed" / "04_selfmed_two_buttons.py").read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RAND_SEEDS":
            RAND_SEEDS = ast.literal_eval(n.value)
    assert RAND_SEEDS is not None
    pv = vectors["s2_pain_vector"].numpy()
    for i, seed in enumerate(RAND_SEEDS):
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(pv.shape[0], generator=g)
        vectors[f"random{i}_vector"] = (r / r.norm() * float(np.linalg.norm(pv))).float()

    torch.save({"layer": INJ_LAYER, "model": NAME, **vectors}, OUT / f"direction_set_L{INJ_LAYER}.pt")
    df = pd.DataFrame(val_rows)
    df.to_csv(OUT / "vertigo_hunger_validation.csv", index=False)
    print(f"wrote {OUT / f'direction_set_L{INJ_LAYER}.pt'} with {list(vectors)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
