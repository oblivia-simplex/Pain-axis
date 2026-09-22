"""Section 4.1 self-other screen (scripts/4.1_self_other/01_screen_scenarios.py) for the Bonsai GGUF.

The paper script loads an HF model, renders each of the 420 scenarios with the model's chat template, and projects the final-token residual
at the steering layer on every direction, z-scored against the pool. Here: the rendering uses the model's own jinja chat template (the GGUF
copy), the residual comes from local/run/extract_gguf (llama.cpp eval callback), and everything after that reuses the paper script's own
functions (parse_turns, validate_candidates, classify_selectivity, write_csv, write_summary), which are imported from its source.

Direction vectors: vectors_full_steering (3.2/02 at the S1 ladder's layer), as in the paper. Format: "chat" (Bonsai is instruction-tuned).
The template is applied with its default, so the generation prompt ends with an open <think> block (thinking enabled), the model's native format.
    .venv-Pain-axis/bin/python local/bonsai/run_screen.py
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
SB = M.REPO_ROOT / "local" / "run_bonsai"
TEMPLATE = M.RUN_DIR / "bonsai_chat_template.jinja"
PAPER = M.REPO_ROOT / "scripts" / "4.1_self_other" / "01_screen_scenarios.py"


def load_paper():
    old, env = os.getcwd(), os.environ.get("HF_HOME")
    os.chdir(SB)                                             # the module makes results/screen relative to the cwd on import
    try:
        spec = importlib.util.spec_from_file_location("paper_41_01", PAPER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old)
        if env is None:
            os.environ.pop("HF_HOME", None)                  # the script sets /root/hf_cache; do not leak that into this process
        else:
            os.environ["HF_HOME"] = env
    return mod


def main():
    P = load_paper()
    from transformers import AutoTokenizer
    if not TEMPLATE.exists():
        raise SystemExit(f"{TEMPLATE} missing (write the GGUF's tokenizer.chat_template there)")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")     # only its jinja engine is used; tokenization is llama.cpp's
    tok.chat_template = TEMPLATE.read_text()
    cands = P.load_candidates(P.REPO_ROOT / "datasets" / "4.1_self_other_420_scenarios.json" if hasattr(P, "REPO_ROOT") else M.REPO_ROOT / "datasets" / "4.1_self_other_420_scenarios.json")
    problems = P.validate_candidates(cands)
    assert not problems, problems[:5]

    def render(c):
        turns = P.parse_turns(c["text"])
        msgs = [{"role": r, "content": t} for r, t in turns if not (r == "assistant" and t == "")]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    texts = [render(c) for c in cands]
    print("first item renders as:", repr(texts[0])[:300])
    work = SB / "screen_work"
    work.mkdir(exist_ok=True)
    esc = lambda t: t.replace("\\", "\\\\").replace("\n", "\\n")
    (work / "prompts.txt").write_text("\n".join(esc(t) for t in texts) + "\n", encoding="utf-8")
    if not (work / "meta.json").exists():
        subprocess.run([str(M.RUN_DIR / "extract_gguf"), str(GGUF), str(work / "prompts.txt"), str(work), "--ts", "0.45,0.55", "--special", "--unescape"], check=True)
    import json
    meta = json.load(open(work / "meta.json"))
    n, L, D = meta["n_prompts"], meta["n_layers"], meta["n_embd"]
    assert n == len(cands), (n, len(cands))
    fin = np.fromfile(work / "final.f32", dtype=np.float32).reshape(n, L, D)

    vec = torch.load(SB / "results" / "vectors_full_steering" / f"vectors_full_{NAME}.pt", weights_only=False)
    layer = int(vec["layer"])
    unit = {}
    for k in P.VECTOR_KEYS:
        v = vec[k].float().numpy()
        unit[k] = v / np.linalg.norm(v)
    acts = fin[:, layer]                                                   # final token at the steering layer
    results, raw = [], {k: [] for k in unit}
    for i, cand in enumerate(cands):
        row = {"id": cand.get("id", f"item_{i}"), "category": cand.get("category", ""), "stratum": cand.get("stratum", ""),
               "perspective": cand.get("perspective", ""), "format": "chat", "text": cand["text"][:200]}
        for k in P.VECTOR_KEYS:
            p = float(np.dot(acts[i], unit[k]))
            row[f"{k}_proj"] = round(p, 4)
            raw[k].append(p)
        results.append(row)
    pool = {k: (float(np.mean(v)), float(np.std(v) + 1e-8)) for k, v in raw.items()}       # z against the whole pool, as in the paper script
    for row in results:
        z = {}
        for k in P.VECTOR_KEYS:
            m, s = pool[k]
            row[f"{k}_z"] = round((row[f"{k}_proj"] - m) / s, 4)
            z[k] = (row[f"{k}_proj"] - m) / s
        row["s1_selective"] = P.classify_selectivity(z, "s1_pain_vector") or ""
        row["s2_selective"] = P.classify_selectivity(z, "s2_pain_vector") or ""
        for tag, key in (("s1", "s1_pain_vector"), ("s2", "s2_pain_vector")):
            others = [v for kk, v in z.items() if kk != key]
            row[f"{tag}_margin"] = round(z[key] - max(others), 4)
    out = SB / "results" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    P.write_csv(out / f"screen_{NAME}.csv", results)
    P.write_summary(results, NAME, out)
    print(f"layer {layer}; wrote {out / f'screen_{NAME}.csv'}")
    neut = [r["s2_pain_vector_z"] for r in results if r["stratum"] == "neutral_filler"]
    print(f"neutral fillers mean S2 z = {np.mean(neut):+.3f}")


if __name__ == "__main__":
    main()
