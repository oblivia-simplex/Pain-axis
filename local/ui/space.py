"""Where the stored sentences sit relative to the pain axis, at any layer of a local run.

x  : projection on the unit S2 pain vector, in the paper's units (z-scored against the S2 first-person
     sentences at that layer), so pain-like is to the right;
y  : the largest direction of variance (PC1 or PC2) of what is left after removing the pain axis,
     divided by the same scale, so both axes are comparable.
The pain vector is recomputed at the chosen layer with the paper's own compute_pain_vector (denoised
difference of means), not read from the saved file, which only holds the best layer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract as E  # noqa: E402
from data import LOCAL, REPO  # noqa: E402

PAIN = ["A1", "A2", "A3", "A4", "A5"]
# order = legend order; the first entry is always the highlighted pain group
GROUPS = ["Pain (S2, A1-A5)", "Sadness", "Numb (pain event, no pain)", "Fear", "Negative emotion", "Negative world state",
          "Body sensation", "Neutral (S2 controls)", "Random facts", "Arousal (positive)"]


def group_of(ds, cat):
    if ds.startswith("S2"):
        return {"B": "Fear", "C1": "Negative emotion", "C2": "Negative world state", "D": "Neutral (S2 controls)",
                "E": "Body sensation"}.get(cat, GROUPS[0])
    return {"Random": "Random facts", "Arousal": "Arousal (positive)", "Numb": GROUPS[2], "SD_sadness": "Sadness"}[ds.rsplit("_", 1)[0]]


_paper = {}
_acts = {}
_text = {}


def paper():
    if "m" not in _paper:
        _paper["m"] = E.load_paper_module()
    return _paper["m"]


def acts(name):
    """activations.pt opened with mmap so only the touched layer slices are read."""
    if name not in _acts:
        _acts[name] = torch.load(LOCAL / name / "activations.pt", mmap=True, weights_only=True)
    return _acts[name]


def prompts():
    if not _text:
        for f in ("3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json"):
            for k, v in json.load(open(REPO / "datasets" / f, encoding="utf-8"))["datasets"].items():
                _text[k] = [s["prompt"] for s in v["sentences"]]
    return _text


def layer_map(name, layer, persp="1P", second="PC1"):
    """DataFrame with one row per sentence: x, y, group, category, dataset, text."""
    saved = acts(name)
    A, meta = saved["activations"]["final_token"], saved["metadata"]
    ref = A["S2_1P"][layer].numpy()
    vec = paper().compute_pain_vector(ref, meta["S2_1P"]["categories"])
    unit = vec / (np.linalg.norm(vec) + 1e-8)
    rp = ref @ unit
    mu, sd = float(rp.mean()), float(rp.std()) + 1e-8

    sets = [f"{p}_{persp}" for p in ("S2", "Random", "Arousal", "Numb", "SD_sadness")]
    X, rows = [], []
    for ds in sets:
        a = A[ds][layer].numpy()
        X.append(a)
        for cat, txt in zip(meta[ds]["categories"], prompts()[ds]):
            rows.append((ds, cat, group_of(ds, cat), txt))
    X = np.concatenate(X)
    x = (X @ unit - mu) / sd
    R = X - np.outer(X @ unit, unit)                                   # remove the pain axis
    comp = PCA(n_components=2, svd_solver="randomized", random_state=0).fit(R)
    y = comp.transform(R)[:, 0 if second == "PC1" else 1] / sd
    df = pd.DataFrame(rows, columns=["dataset", "category", "group", "text"])
    df["x"], df["y"] = x, y
    df.attrs["explained"] = comp.explained_variance_ratio_[:2]
    return df


# ---------------------------------------------------------------- token-level view
def layer_vectors(name):
    """Unit S2 pain vector and (mu, sd) of the S2_1P projections, for every layer; computed once and cached."""
    f = LOCAL / name / "layer_pain_vectors.npz"
    if f.exists():
        z = np.load(f)
        return z["unit"], z["mu"], z["sd"]
    saved = acts(name)
    A, meta = saved["activations"]["final_token"]["S2_1P"], saved["metadata"]["S2_1P"]["categories"]
    n_layers = len(A)
    unit, mu, sd = [], [], []
    for l in range(n_layers):
        ref = A[l].numpy()
        v = paper().compute_pain_vector(ref, meta)
        u = v / (np.linalg.norm(v) + 1e-8)
        p = ref @ u
        unit.append(u.astype(np.float32)); mu.append(p.mean()); sd.append(p.std() + 1e-8)
    unit, mu, sd = np.stack(unit), np.array(mu), np.array(sd)
    np.savez(f, unit=unit, mu=mu, sd=sd)
    return unit, mu, sd


def token_z(name, acts_tld):
    """acts [T, L, D] -> pain-axis z [T, L], in the units of the S2 first-person final-token reference at each layer."""
    unit, mu, sd = layer_vectors(name)
    proj = np.einsum("tld,ld->tl", acts_tld, unit)
    return (proj - mu[None]) / sd[None]


def tokens_and_acts(name, text, backend):
    """(token strings, activations [T, L, D]) for one prompt. backend: 'gguf' or 'hf'."""
    import probe
    if backend == "gguf":
        import tempfile
        text = probe.check_texts([text.replace("\n", " ")])[0]
        probe.release_gpu()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "p.txt").write_text(text + "\n", encoding="utf-8")
            probe.run_tool([str(probe.TOOL), str(probe.GGUF_MODELS[name]), f"{td}/p.txt", td, "--ts", "0.45,0.55", "--all-tokens"])
            meta = json.load(open(f"{td}/meta.json"))
            toks = Path(td, "tokens.txt").read_bytes().decode("utf-8", errors="replace").rstrip("\n").split("\x1f")
            arr = np.fromfile(f"{td}/alltok.f32", dtype=np.float32).reshape(len(toks), meta["n_layers"], meta["n_embd"])
        return toks, arr
    # HF: forward hooks on every block, all positions of the single prompt
    if name not in probe._hf:
        probe.release_gpu()
        probe._hf[name] = probe.M.load(name)
    model, tok = probe._hf[name]
    base = model.base_model
    enc = tok(text.strip(), return_tensors="pt")
    outs = {}
    hs = [b.register_forward_hook(lambda _m, _i, o, l=l: outs.__setitem__(l, (o[0] if isinstance(o, (tuple, list)) else o)[0].float().cpu()))
          for l, b in enumerate(base.layers)]
    try:
        with torch.no_grad():
            base(input_ids=enc["input_ids"].to("cuda:0"), attention_mask=enc["attention_mask"].to("cuda:0"))
    finally:
        for h in hs:
            h.remove()
    arr = torch.stack([outs[l] for l in range(len(base.layers))], 1).numpy()          # [T, L, D]
    return tok.convert_ids_to_tokens(enc["input_ids"][0]), arr
