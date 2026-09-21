"""Section 3.2 for a GGUF model (Ternary Bonsai 2 27B) that has no HF weights.

1. writes every prompt of the paper's datasets, one per line, in extract.py's order;
2. runs local/run/extract_gguf (built from extract_gguf.cpp against the PrismML llama.cpp fork),
   which captures the residual stream after every block through llama.cpp's eval callback;
3. packs its output into the paper's activations.pt layout;
4. calls the paper's own process_model() on it, exactly as extract.py does.

Build the tool first (see local/README.md), then:
    .venv-Pain-axis/bin/python local/bonsai/run_bonsai.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402
from extract import load_datasets, load_paper_module, setup_run_dir  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
TOOL = M.RUN_DIR / "extract_gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=GGUF)
    ap.add_argument("--ts", default="0.45,0.55", help="tensor split over the two GPUs")
    args = ap.parse_args()

    setup_run_dir()
    paper = load_paper_module()
    dataset = load_datasets()
    out_dir = paper.OUTPUT_DIR / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.json").exists():
        print("already done")
        return

    raw = out_dir / "gguf_raw"
    raw.mkdir(exist_ok=True)
    sets = list(dataset["datasets"].items())
    prompts = [s["prompt"] for _, ds in sets for s in ds["sentences"]]
    assert all("\n" not in p for p in prompts), "prompt with a newline would break the one-per-line file"
    (raw / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    if not (raw / "meta.json").exists():
        subprocess.run([str(TOOL), str(args.gguf), str(raw / "prompts.txt"), str(raw), "--ts", args.ts], check=True)
    meta = json.load(open(raw / "meta.json"))
    n, L, D = meta["n_prompts"], meta["n_layers"], meta["n_embd"]
    assert n == len(prompts), (n, len(prompts))
    print(f"{n} prompts, {L} layers, d_model {D}")

    acts = {"final_token": {}, "mean": {}}
    md = {}
    for kind, fname in (("final_token", "final.f32"), ("mean", "mean.f32")):
        arr = np.fromfile(raw / fname, dtype=np.float32).reshape(n, L, D)
        i = 0
        for ds_name, ds in sets:
            k = len(ds["sentences"])
            acts[kind][ds_name] = {l: torch.from_numpy(np.ascontiguousarray(arr[i:i + k, l])) for l in range(L)}
            i += k
        del arr
    for ds_name, ds in sets:
        md[ds_name] = {"categories": [s["category"] for s in ds["sentences"]],
                       "sets": [s["set"] for s in ds["sentences"]]}
    torch.save({"activations": acts, "metadata": md, "layers": list(range(L)), "model_name": str(args.gguf.name),
                "n_layers": L, "d_model": D}, out_dir / "activations.pt")
    del acts

    summary = paper.process_model(args.gguf.name, NAME, dataset, out_dir)
    print(f"best layers: final_token={summary['best_layer_final_token']} mean={summary['best_layer_mean']}")


if __name__ == "__main__":
    main()
