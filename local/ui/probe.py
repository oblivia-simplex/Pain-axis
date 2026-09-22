"""Score arbitrary text on a local run's pain vector.

Same readout as the paper's z-scores: final-token residual at the model's best layer, projected on
the unit S2 pain vector, z-scored against the S2 first-person sentences (mean/std of their projections).

Backends
  HF models in models.REGISTRY  loaded once per process (bf16, CPU overflow), see load_hf()
  Bonsai GGUF                   local/run/extract_gguf, one subprocess per call (model reload ~4 s when
                                the file is in the page cache)
Only one big model should be resident on the GPUs at a time: call release_gpu() before switching.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402
import extract as E  # noqa: E402
from data import LOCAL  # noqa: E402

GGUF_MODELS = {"Bonsai_2_27B_ternary": Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"}
TOOL = M.RUN_DIR / "extract_gguf"
PAIN = ["A1", "A2", "A3", "A4", "A5"]

_hf = {}          # name -> (model, tokenizer); at most one entry


def can_probe(name):
    if name in GGUF_MODELS:
        return TOOL.exists() and GGUF_MODELS[name].exists()
    return name in M.REGISTRY


def load_reference(name):
    """Unit S2 vector, layer, and the S2_1P projection statistics for z-scoring."""
    d = LOCAL / name
    layer = json.load(open(d / "summary.json"))["best_layer_final_token"]
    v = torch.load(d / "final_token" / "pain_vectors.pt", weights_only=False)["s2_pain_vector"].float().numpy()
    unit = v / (np.linalg.norm(v) + 1e-8)
    saved = torch.load(d / "activations.pt", mmap=True, weights_only=True)    # only this layer's slice is read
    acts = saved["activations"]["final_token"]["S2_1P"][layer].numpy()
    cats = np.array(saved["metadata"]["S2_1P"]["categories"])
    proj = acts @ unit
    mu, sd = float(proj.mean()), float(proj.std())
    z = (proj - mu) / (sd + 1e-8)
    return {"unit": unit, "layer": int(layer), "mu": mu, "sd": sd,
            "z_pain": z[np.isin(cats, PAIN)], "z_ctrl": z[~np.isin(cats, PAIN)]}


MAX_CHARS = 1200      # ~300-400 tokens; extract_gguf's default context is 512 tokens


def check_texts(texts):
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        raise ValueError("enter at least one prompt")
    if any(len(t) > MAX_CHARS for t in texts):
        raise ValueError(f"a prompt is longer than {MAX_CHARS} characters (the GGUF tool reads at most 512 tokens)")
    return texts


def run_tool(cmd, **kw):
    """subprocess.run that reports the tool's own stderr instead of just an exit status."""
    r = subprocess.run(cmd, capture_output=True, timeout=600, **kw)
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"{Path(cmd[0]).name} exited with {r.returncode}: " + " | ".join(tail))
    return r


def release_gpu():
    _hf.clear()
    try:
        import chat                       # a running chat server also holds the GPUs
        chat.stop()
    except ImportError:
        pass
    M.free_gpu()


def _acts_hf(name, texts, layer):
    if name not in _hf:
        release_gpu()
        _hf[name] = M.load(name)
    model, tok = _hf[name]
    fin, _ = E.extract(model, tok, texts, batch_size=8, bos="tokenizer")
    return fin[layer].numpy()


def _acts_gguf(name, texts, layer):
    release_gpu()                                   # the tool needs the VRAM
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "p.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        run_tool([str(TOOL), str(GGUF_MODELS[name]), f"{td}/p.txt", td, "--ts", "0.45,0.55"])
        meta = json.load(open(f"{td}/meta.json"))
        arr = np.fromfile(f"{td}/final.f32", dtype=np.float32).reshape(meta["n_prompts"], meta["n_layers"], meta["n_embd"])
        return arr[:, layer].copy()


def score(name, texts, ref):
    """z-score of each text on the pain vector (higher = more pain-like, in the paper's units)."""
    texts = check_texts(texts)
    acts = (_acts_gguf if name in GGUF_MODELS else _acts_hf)(name, texts, ref["layer"])
    return texts, (acts @ ref["unit"] - ref["mu"]) / (ref["sd"] + 1e-8)
