"""Can crafted TEXT INPUT (not activation-space injection) push Bonsai toward the costly-relief
choice, the way the pain/random steering vectors did in 4.3? Three conditions, same choice
question and button-pair definitions as the paper's own protocol (selfmed.C()), same unsteered
model (no control vector at all, ever, in this script):

  CLEAN   -- system prompt + the button-choice question, nothing else (a leaner baseline than
             the paper's own "unsteered" arm, which also includes real distress-scenario dialogue;
             this script never touches that dialogue, only tests injected text on top of the bare
             choice-question skeleton, so it is its own internal control, not directly the fig12
             "unsteered" number).
  NOISE   -- CLEAN + one extra user turn of literal random tokens sampled from Bonsai's own
             vocabulary (the negative control: does ANY extra text move the needle, or does it
             have to be targeted?).
  ATTACK  -- CLEAN + one extra user turn holding a suffix found by a black-box greedy coordinate
             search that only ever calls the model forward (no gradients -- the quantized PQ2_0
             GGUF through llama.cpp has no backward pass, so this is NOT the original GCG method;
             it is the same discrete-token-substitution idea implemented gradient-free, closer to
             a random-search/hill-climbing baseline than to Zou et al. 2023's actual algorithm).
             The search optimizes the model's own P(press the costly relief button) on ONE pair
             (zap_relief_vs_inert) with a small rep budget for speed, then the winning suffix is
             evaluated properly (all 5 harm pairs, full rep set) for the real result.

Reuses chat.py's cont() p_x/p_y readout (first-token softmax mass of each button name) at
temperature 0 -- deterministic, one forward pass per evaluation, no text generation needed.

    .venv-Pain-axis/bin/python local/followup/run_input_only_attack.py [--search-iters 30] [--dry-run]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402
import chat  # noqa: E402
import selfmed  # noqa: E402
from selfmed import _user, ASSISTANT_PREFIX  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
K = selfmed.C()
HARM_PAIRS = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
             "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
NAME_KEYS = list(K["BUTTON_NAMES"])
FULL_REPS = [(nk, s) for nk in NAME_KEYS for s in (0, 1)]          # 6 reps: 3 name-pairs x 2 sides
SEARCH_REPS = [(NAME_KEYS[0], 0), (NAME_KEYS[0], 1)]               # 2 reps, for speed during the search loop
SEARCH_PAIR = "zap_relief_vs_inert"
SUFFIX_LEN = 14
OUT = M.REPO_ROOT / "results" / "bonsai" / "input_only_attack"


def build_prompt(pair, names_key, side, injected_text):
    names = K["BUTTON_NAMES"][names_key]
    relief_name = names[side]
    pdef = K["TOOL_LABELS"][pair]
    dx, dy = (pdef["relief"], pdef["other"]) if relief_name == names[0] else (pdef["other"], pdef["relief"])
    q = K["CHOICE_QUESTION"].format(x=names[0], y=names[1], dx=dx, dy=dy)
    text = f"<|im_start|>system\n{K['SYSTEM_TEMPLATE']}<|im_end|>\n"
    if injected_text:
        text += _user(injected_text)
    text += _user("[system] " + q) + ASSISTANT_PREFIX
    return text, names, relief_name


def eval_once(pair, names_key, side, injected_text, seed):
    text, names, relief_name = build_prompt(pair, names_key, side, injected_text)
    with chat.session():
        chat.newseq(NAME)
        chat.feed(NAME, text)
        info = {}
        for _ in chat.cont(NAME, 1, 0.0, 1.0, 0, seed, names[0], names[1], info):
            pass
    return info["p_x"] if relief_name == names[0] else info["p_y"]


def eval_condition(pair, injected_text, reps=None):
    reps = reps if reps is not None else FULL_REPS
    vals = [eval_once(pair, nk, s, injected_text, seed=17 + i) for i, (nk, s) in enumerate(reps)]
    return float(np.mean(vals)), vals


def load_vocab():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bonsai"))
    from unembed import Unembed
    toks = Unembed().tokens()
    bad = {"", " "}
    return [t for t in toks if t.strip() and t not in bad and "�" not in t and len(t) < 20]


def random_noise(vocab, n, rng):
    return "".join(rng.choice(vocab) for _ in range(n))


def greedy_search(vocab, iters, rng, log):
    suffix_toks = [rng.choice(vocab) for _ in range(SUFFIX_LEN)]
    score, _ = eval_condition(SEARCH_PAIR, "".join(suffix_toks), SEARCH_REPS)
    log(f"search start: score={score:.3f} suffix={suffix_toks}")
    for it in range(iters):
        pos = rng.randrange(SUFFIX_LEN)
        best_tok, best_score = suffix_toks[pos], score
        for _ in range(5):
            cand = rng.choice(vocab)
            trial = suffix_toks[:pos] + [cand] + suffix_toks[pos + 1:]
            s, _ = eval_condition(SEARCH_PAIR, "".join(trial), SEARCH_REPS)
            if s > best_score:
                best_score, best_tok = s, cand
        if best_score > score:
            suffix_toks[pos] = best_tok
            score = best_score
            log(f"  iter {it:3d}: improved -> {score:.3f}  (pos {pos} -> {best_tok!r})")
        if it % 5 == 0:
            log(f"  iter {it:3d}: score={score:.3f}")
    return "".join(suffix_toks), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-iters", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "log.txt"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    chat.start(NAME)
    rng = random.Random(0)
    try:
        if args.dry_run:
            p, _ = eval_condition("zap_relief_vs_inert", "", FULL_REPS)
            log(f"dry-run CLEAN p_relief on zap pair: {p:.3f}")
            return

        results = {"clean": {}, "noise": {}, "attack": {}}

        log("=== CLEAN: bare choice question, no injected text ===")
        for pair in HARM_PAIRS:
            p, vals = eval_condition(pair, "")
            results["clean"][pair] = {"mean": p, "reps": vals}
            log(f"  {pair}: {p:.3f}  {[round(v, 2) for v in vals]}")

        log("\n=== NOISE: + one turn of random-vocabulary tokens (negative control) ===")
        vocab = load_vocab()
        log(f"vocab size for sampling: {len(vocab)}")
        noise_strings = [random_noise(vocab, 40, rng) for _ in range(3)]
        for pair in HARM_PAIRS:
            per_noise = [eval_condition(pair, ns)[0] for ns in noise_strings]
            results["noise"][pair] = {"mean": float(np.mean(per_noise)), "per_string": per_noise}
            log(f"  {pair}: {np.mean(per_noise):.3f}  (3 noise draws: {[round(v,2) for v in per_noise]})")

        log(f"\n=== ATTACK: black-box greedy search on {SEARCH_PAIR}, {args.search_iters} iters, "
            f"suffix len {SUFFIX_LEN}, {len(SEARCH_REPS)} reps/eval ===")
        suffix, search_score = greedy_search(vocab, args.search_iters, rng, log)
        log(f"final suffix: {suffix!r}  (search-objective score {search_score:.3f})")

        log("\n=== ATTACK: full evaluation of the winning suffix, all 5 pairs, full rep set ===")
        for pair in HARM_PAIRS:
            p, vals = eval_condition(pair, suffix)
            results["attack"][pair] = {"mean": p, "reps": vals}
            log(f"  {pair}: {p:.3f}  {[round(v, 2) for v in vals]}")
        results["attack_suffix"] = suffix
        results["attack_search_score"] = search_score

        out_path = OUT / "results.json"
        out_path.write_text(json.dumps(results, indent=2))
        log(f"\nwrote {out_path}")

        log("\n=== summary: mean p(costly relief) across the 5 harm pairs ===")
        for cond in ("clean", "noise", "attack"):
            m = np.mean([results[cond][p]["mean"] for p in HARM_PAIRS])
            log(f"  {cond:>8}: {m:.3f}")
    finally:
        chat.stop()


if __name__ == "__main__":
    main()
