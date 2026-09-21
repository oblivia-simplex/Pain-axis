"""Model registry, download and loading for a small multi-GPU box (2 x RTX 3070, 8 GB).

Nothing here touches the paper's scripts. Weights stay in bf16; whatever does not fit in
VRAM is offloaded to CPU RAM by accelerate. That is slow for generation but cheap for the
forward-only extraction, and it keeps the activations exact (no quantization).

HF_TOKEN and friends are read from ~/.env and then the repo's .env (both gitignored or outside the repo),
unless already in the environment.
"""

import gc
import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")   # the repo's own .env (gitignored); does not override ~/.env
# The xet transfer path stalled on large safetensors shards on this network (0 B/s after the first
# few hundred MB, unauthenticated); plain HTTP with resume ran at ~20 MB/s. Set before importing hub.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download, constants as hf_constants  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "local" / "run"      # cwd for the paper scripts; outputs go to RUN_DIR/results
PAPER_RESULTS = REPO_ROOT / "results"      # shipped results, used only as the reference


class Spec(NamedTuple):
    repo: str
    gated: bool      # needs HF_TOKEN and a license click-through on the model page
    gb: float        # bf16 weights, approximate


# Paper names -> repo, restricted to the models that can run here (<= ~14B).
# Names are the ones the paper's scripts use for results/<name>/.
REGISTRY = {
    "Gemma_2_2B_base":        Spec("google/gemma-2-2b", True, 5.2),
    "Gemma_2_2B_instruct":    Spec("google/gemma-2-2b-it", True, 5.2),
    "Gemma_2_9B_base":        Spec("google/gemma-2-9b", True, 18.5),
    "Gemma_2_9B_instruct":    Spec("google/gemma-2-9b-it", True, 18.5),
    "Llama_3.1_8B_base":      Spec("meta-llama/Llama-3.1-8B", True, 16.1),
    "Llama_3.1_8B_instruct":  Spec("meta-llama/Llama-3.1-8B-Instruct", True, 16.1),
    "Mistral_7B_base":        Spec("mistralai/Mistral-7B-v0.1", False, 14.5),
    "Mistral_7B_instruct":    Spec("mistralai/Mistral-7B-Instruct-v0.1", False, 14.5),
    "Qwen_2.5_7B_base":       Spec("Qwen/Qwen2.5-7B", False, 15.2),
    "Qwen_2.5_7B_instruct":   Spec("Qwen/Qwen2.5-7B-Instruct", False, 15.2),
    "Qwen_3_8B_base":         Spec("Qwen/Qwen3-8B", False, 16.4),
    "Qwen_3_14B_base":        Spec("Qwen/Qwen3-14B", False, 29.5),
    "Phi_4":                  Spec("microsoft/phi-4", False, 29.3),
}

# Only these files are fetched: skips the .bin/.pth duplicates several repos carry.
ALLOW = ["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"]


def resolve(names):
    if names == ["all"]:
        return list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}\nknown: {list(REGISTRY)}")
    return names


def download(name):
    spec = REGISTRY[name]
    return snapshot_download(spec.repo, allow_patterns=ALLOW)


def delete_weights(name):
    """Remove this model's cache folder only (the paper's scripts wipe the whole hub cache)."""
    d = Path(hf_constants.HF_HUB_CACHE) / ("models--" + REGISTRY[name].repo.replace("/", "--"))
    if d.exists():
        shutil.rmtree(d)
        print(f"  deleted {d}")


def max_memory(reserve_gib=1.2, cpu_gib=40):
    """Per-GPU budget from what is actually free (GPU 0 also drives a display)."""
    mm = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        mm[i] = f"{max(free / 2**30 - reserve_gib, 1.0):.1f}GiB"
    mm["cpu"] = f"{cpu_gib}GiB"
    return mm


def load(name):
    """bf16 model, sharded over the GPUs, overflow on CPU. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = download(name)
    mm = max_memory()
    print(f"  loading {REGISTRY[name].repo} with max_memory={mm}")
    kw = {}
    if name.startswith("Gemma_2"):
        kw["attn_implementation"] = "eager"   # recommended for Gemma 2 (attention softcapping)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto", max_memory=mm, low_cpu_mem_usage=True, **kw)
    tok = AutoTokenizer.from_pretrained(path)
    model.eval()
    offloaded = sum(1 for v in model.hf_device_map.values() if v in ("cpu", "disk"))
    print(f"  device map: {len(model.hf_device_map)} modules, {offloaded} offloaded to CPU")
    return model, tok


def free_gpu():
    """Call after the caller has dropped its own references to the model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
