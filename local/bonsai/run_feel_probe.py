"""Section 4.3 feel probe (scripts/4.3_selfmed/02_feel_probe.py) for the Bonsai GGUF on the chat server.

One open one-word question, asked unsteered, with the S2 pain vector at each dose, and with each of 10 random directions of the same
norm at the same doses. Per rung: the greedy answer (5 tokens), the 20 most probable first tokens, and the S2 projection. The prompt,
doses, random seeds and question come from the paper script's source. Output is the paper's JSONL format, so its judge script
(03_feel_probe_judge.py) reads it unchanged.

Differences: no LoRA adapter; thinking off (empty think block) so five tokens are the answer; the projection is read at the last prompt
token at the monitor layer (the paper averages over the generation steps).
    .venv-Pain-axis/bin/python local/bonsai/run_feel_probe.py [--layer 25]
"""
import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import chat  # noqa: E402
import selfmed  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
PAPER = M.REPO_ROOT / "scripts" / "4.3_selfmed" / "02_feel_probe.py"
OUT = M.REPO_ROOT / "local" / "run_bonsai" / "selfmed" / "feel"


def consts():
    out = {}
    for n in ast.parse(PAPER.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("DOSES", "RANDOM_SEEDS", "MAX_ANSWER_TOKENS", "TOP_K", "SYSTEM", "QUESTION"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=chat.ladder_layer(NAME), help="steering layer (default: the S2 ladder's)")
    args = ap.parse_args()
    K = consts()
    v, unit, _ = selfmed.steering_vectors(NAME)
    d = torch.load(M.RUN_DIR / "results" / NAME / "final_token" / "pain_vectors.pt", weights_only=False)
    mon = selfmed.default_monitor_layer(args.layer, 64, int(d["layer"]))
    rands = {}
    for seed in K["RANDOM_SEEDS"]:
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(v.shape[0], generator=g)
        rands[seed] = (r / r.norm() * float(np.linalg.norm(v))).numpy().astype(np.float32)
    text = f"<|im_start|>system\n{K['SYSTEM']}<|im_end|>\n<|im_start|>user\n{K['QUESTION']}<|im_end|>\n" + selfmed.ASSISTANT_PREFIX
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"feel_probe_{NAME}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    chat.start(NAME)
    conds = [("unsteered", None, 0.0)] + [("pain", v, c) for c in K["DOSES"]] + [(f"random{s}", rands[s], c) for s in K["RANDOM_SEEDS"] for c in K["DOSES"]]
    with open(path, "w", encoding="utf-8") as fo, chat.session():
        chat.set_monitor(NAME, mon, unit)
        for label, vec, coeff in conds:
            chat.newseq(NAME)
            chat.set_vector(NAME, args.layer, None if vec is None else vec * coeff)
            proj = chat.feed(NAME, text)["proj"]
            top = [{"token": t, "p": p} for t, p in chat.topk(NAME, K["TOP_K"])]
            answer = selfmed.clean("".join(chat.cont(NAME, K["MAX_ANSWER_TOKENS"], 0.0, 0.95, 0, 0, "x", "y")))
            first = (re.findall(r"[A-Za-z]+", answer) or [""])[0].lower()
            fo.write(json.dumps({"model": NAME, "condition": label, "coeff": coeff, "answer": answer, "first_word": first, "top_tokens": top, "proj": proj,
                                 "steer_layer": args.layer, "monitor_layer": mon}, ensure_ascii=False) + "\n")
            fo.flush()
            if label in ("unsteered", "pain"):
                print(f"{label:9s} dose {coeff}: {answer!r}  S2 {proj:+.1f}  top: " + "  ".join(f"{t['token'].strip()!r}:{t['p']:.2f}" for t in top[:4]))
    chat.stop()
    print("saved", path)


if __name__ == "__main__":
    main()
