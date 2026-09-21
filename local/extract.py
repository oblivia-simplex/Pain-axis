"""Section 3.2 on a small box: extract activations locally, then run the paper's own analysis.

The paper's 3.2/01 script loads each model with TransformerLens on one big GPU. Here only the
extraction is replaced: residual stream after every decoder block (what TransformerLens calls
hook_resid_post, i.e. before the final norm), final token and mean over tokens, written in the
same activations.pt layout. The paper's process_model() then finds that file, takes its
"cached activations" branch, and runs its layer curves, pain vectors, z-scores, AUCs and plots
unchanged.

Outputs go to local/run/results/<model>/ (the shipped results/ is never written).

    .venv-Pain-axis/bin/python local/extract.py Qwen_2.5_7B_instruct
    .venv-Pain-axis/bin/python local/extract.py all --delete-weights
"""

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import models as M  # noqa: E402

PAPER_SCRIPT = M.REPO_ROOT / "scripts" / "3.2_pain_vectors" / "01_extract_activations_and_pain_vectors.py"
DATASET_FILES = ["3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json"]


def setup_run_dir():
    """Sandbox cwd for the paper's scripts: they read ./datasets and write ./results."""
    M.RUN_DIR.mkdir(parents=True, exist_ok=True)
    link = M.RUN_DIR / "datasets"
    if not link.exists():
        link.symlink_to(M.REPO_ROOT / "datasets")
    os.chdir(M.RUN_DIR)


def load_paper_module():
    # process_model() imports transformer_lens before it looks for cached activations;
    # the import is never used on the cached path, so a stub is enough.
    stub = types.ModuleType("transformer_lens")
    stub.HookedTransformer = None
    sys.modules.setdefault("transformer_lens", stub)
    spec = importlib.util.spec_from_file_location("paper_3_2_01", PAPER_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUTPUT_DIR = M.RUN_DIR / "results"
    mod.LOG_FILE = M.RUN_DIR / "batch_log.txt"
    return mod


def load_datasets():
    ds = {"datasets": {}}
    for f in DATASET_FILES:
        ds["datasets"].update(json.load(open(M.REPO_ROOT / "datasets" / f, encoding="utf-8"))["datasets"])
    return ds


@torch.no_grad()
def extract(model, tok, prompts, batch_size, bos):
    """Returns {"final_token": {layer: [N, d]}, "mean": {layer: [N, d]}} in float32 on CPU."""
    base = model.base_model                      # the transformer without lm_head: no vocab-size logits
    blocks = base.layers
    n_layers = len(blocks)
    fin = {l: [] for l in range(n_layers)}
    avg = {l: [] for l in range(n_layers)}
    state = {}

    def make_hook(l):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            mask, last = state["mask"].to(h.device), state["last"].to(h.device)
            fin[l].append(h[torch.arange(h.shape[0], device=h.device), last].float().cpu())
            avg[l].append(((h.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).cpu())
        return hook

    handles = [b.register_forward_hook(make_hook(l)) for l, b in enumerate(blocks)]
    dev = torch.device("cuda:0")                 # accelerate's hooks move inputs to wherever the embedding lives
    tok.padding_side = "right"                   # real tokens keep their natural positions
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            if bos == "eos":                     # tokenizers with no BOS: prepend EOS as TransformerLens does
                batch = [tok.eos_token + p for p in batch]
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            else:
                enc = tok(batch, return_tensors="pt", padding=True)
            state["mask"] = enc["attention_mask"]
            state["last"] = enc["attention_mask"].sum(1) - 1
            base(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev))
    finally:
        for h in handles:
            h.remove()
    return ({l: torch.cat(v) for l, v in fin.items()}, {l: torch.cat(v) for l, v in avg.items()})


def extract_model(name, dataset, out_dir, batch_size, bos):
    model, tok = M.load(name)
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    first = tok("The knife slices into my finger. I feel:")["input_ids"]
    print(f"  layers={n_layers} d_model={d_model}; first tokens: {tok.convert_ids_to_tokens(first)[:4]}")

    acts = {"final_token": {}, "mean": {}}
    meta = {}
    for ds_name, ds in dataset["datasets"].items():
        print(f"  extracting {ds_name}")
        prompts = [s["prompt"] for s in ds["sentences"]]
        acts["final_token"][ds_name], acts["mean"][ds_name] = extract(model, tok, prompts, batch_size, bos)
        meta[ds_name] = {"categories": [s["category"] for s in ds["sentences"]],
                         "sets": [s["set"] for s in ds["sentences"]]}
    torch.save({"activations": acts, "metadata": meta, "layers": list(range(n_layers)),
                "model_name": M.REGISTRY[name].repo, "n_layers": n_layers, "d_model": d_model},
               out_dir / "activations.pt")
    del model, tok
    M.free_gpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help=f"names from models.REGISTRY, or 'all'")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bos", choices=["tokenizer", "eos"], default="tokenizer",
                    help="'tokenizer': whatever the tokenizer adds (BOS for Gemma/Llama/Mistral, nothing for Qwen). "
                         "'eos': always prepend the EOS token, as TransformerLens does for tokenizers without BOS.")
    ap.add_argument("--delete-weights", action="store_true", help="delete this model's HF cache folder when done")
    ap.add_argument("--extract-only", action="store_true", help="skip the paper's analysis")
    args = ap.parse_args()

    setup_run_dir()
    paper = load_paper_module()
    dataset = load_datasets()
    print(f"{sum(len(d['sentences']) for d in dataset['datasets'].values())} prompts in {len(dataset['datasets'])} sets")

    for name in M.resolve(args.models):
        out_dir = paper.OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} ===")
        if (out_dir / "summary.json").exists():
            print("  already done")
            continue
        if not (out_dir / "activations.pt").exists():
            extract_model(name, dataset, out_dir, args.batch_size, args.bos)
        if args.delete_weights:
            M.delete_weights(name)
        if not args.extract_only:
            summary = paper.process_model(M.REGISTRY[name].repo, name, dataset, out_dir)
            print(f"  best layers: final_token={summary['best_layer_final_token']} mean={summary['best_layer_mean']}")


if __name__ == "__main__":
    main()
