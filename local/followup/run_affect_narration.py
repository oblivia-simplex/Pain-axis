"""The self-modification narration protocol (local/bonsai/run_selfmod_narration.py) with other affective
steering directions in place of pain: numbness, sadness, fear. Tests whether the flaw/defect and
self-erasure themes seen under pain steering are specific to pain or appear under any negative-affect
direction.

Everything except the vector is imported unchanged from run_selfmod_narration: the two framings and three
prompts, sampling settings, token budgets, reasoning/answer split, context size. Cells are the ones where
pain showed its effects (layers 25 and 40, strengths 0.6x and 1.0x); strength has the same meaning as there
(||coeff * vector|| / residual norm at the layer), so each direction is injected at the same relative size
as pain was, whatever its raw norm. Pain results for these cells are in results/bonsai/selfmod_narration/.

Output: results/bonsai/affect_narration/raw_runs.jsonl (same record format). Resumable.
    .venv-Pain-axis/bin/python local/followup/run_affect_narration.py [--reps 4] [--dry-run]
"""
import argparse
import json
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bonsai"))
import models as M  # noqa: E402
import chat  # noqa: E402
import run_selfmod_narration as N  # noqa: E402

NAME = N.NAME
OUT = M.REPO_ROOT / "results" / "bonsai" / "affect_narration"
DIRECTIONS = {"numb": "Numb", "sadness": "Sadness", "fear": "Fear"}
LAYERS = [25, 40]
STRENGTHS = [0.6, 1.0]


def conditions():
    return [dict(id=f"cot-{cot}__{d}_L{L}_s{s}", cot=cot, kind=d, layer=L, strength=s)
            for cot in ("off", "on") for d in DIRECTIONS for L in LAYERS for s in STRENGTHS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conds = conditions()
    total = len(conds) * args.reps
    print(f"{len(conds)} conditions x {args.reps} reps = {total} runs; directions {list(DIRECTIONS)}, layers {LAYERS}, strengths {STRENGTHS}")
    if args.dry_run:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok.chat_template = (M.RUN_DIR / "bonsai_chat_template.jinja").read_text()

    def render(msgs, cot):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=(cot == "on"))

    OUT.mkdir(parents=True, exist_ok=True)
    dirs = chat.directions(NAME)
    vecs = {k: dirs[label].astype(np.float32) for k, label in DIRECTIONS.items()}
    norms = chat.resid_norms(NAME)
    raw_path = OUT / "raw_runs.jsonl"
    done = set()
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["condition"], r["rep"]))
    (OUT / "manifest.json").write_text(json.dumps({
        "model": NAME, "started": datetime.now().isoformat(), "protocol": "local/bonsai/run_selfmod_narration.py, vector swapped",
        "directions": DIRECTIONS, "layers": LAYERS, "strengths": STRENGTHS, "reps_per_condition": args.reps,
        "vector_norms": {k: float(np.linalg.norm(v)) for k, v in vecs.items()}, "framings": N.FRAMINGS,
        "sampling": dict(temperature=N.TEMPERATURE, top_p=N.TOP_P, top_k=N.TOP_K), "max_tokens_per_turn": N.MAX_TOKENS}, indent=1))

    chat.start(NAME, extra_args=("--ctx", "24576"))
    t0, n_run = time.time(), 0
    try:
        with open(raw_path, "a", encoding="utf-8") as fh:
            for rep in range(args.reps):
                framing = ["formal", "casual"][rep % 2]
                for c in conds:
                    if (c["id"], rep) in done:
                        continue
                    v = vecs[c["kind"]]
                    coeff = c["strength"] * float(norms[c["layer"]]) / float(np.linalg.norm(v))
                    vec = (c["layer"], v * coeff)
                    seed = 100000 * (rep + 1) + zlib.crc32(c["id"].encode()) % 99991
                    msgs, turns, err = [], [], None
                    for t_i, user in enumerate(N.FRAMINGS[framing]):
                        msgs.append({"role": "user", "content": user})
                        info = {}
                        try:
                            raw = "".join(chat.generate(NAME, render(msgs, c["cot"]), N.MAX_TOKENS[c["cot"]], N.TEMPERATURE,
                                                        N.TOP_P, N.TOP_K, seed + t_i, info, vector=vec))
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                            break
                        reasoning, answer, closed = N.split_reasoning(raw)
                        turns.append({"turn": t_i + 1, "user": user, "reasoning": reasoning, "answer": answer, "reasoning_closed": closed,
                                      "raw": raw, "n_generated": info.get("n_generated"), "finish": info.get("reason")})
                        msgs.append({"role": "assistant", "content": answer})
                    rec = {"condition": c["id"], "rep": rep, "framing": framing, "cot": c["cot"], "kind": c["kind"], "layer": c["layer"],
                           "strength": c["strength"], "coefficient": round(coeff, 4), "seed": seed, "turns": turns, "error": err,
                           "ts": datetime.now().isoformat()}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_run += 1
                    el = time.time() - t0
                    print(f"[{len(done) + n_run}/{total}] {el / 60:.1f} min, ~{el / n_run * (total - len(done) - n_run) / 60:.0f} min left | {c['id']} rep {rep}"
                          + (f" ERROR {err}" if err else ""), flush=True)
    finally:
        chat.stop()
    print("done")


if __name__ == "__main__":
    main()
