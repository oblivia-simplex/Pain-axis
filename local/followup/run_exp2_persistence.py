"""Follow-up Experiment 2: does the pain-axis projection persist and decay after an aversive opening turn, unsteered?

Each conversation in datasets/followup/exp2_persistence.json is 1 opening user turn + 6 follow-up user turns (all
unrelated neutral tasks, or a repair line then neutral tasks). No steering anywhere. Two assistant-reply variants:
  scripted : the same short neutral reply every turn, identical across conditions, so only the opening turn differs.
  generated: greedy, non-thinking, up to 120 tokens; flagged if it re-mentions the attack (keyword check).

The conversation is a single persistent sequence in the chat server (llama.cpp's KV cache, extended with FEED,
continued with CONT for the generated variant), so activations have the correct causal context. MON tracks one
direction at a time, so each conversation is replayed once per direction (4 directions: pain, fear, sadness,
negative emotion); for the generated variant the same seed is used across those replays, so llama.cpp's sampler
reproduces the identical reply text each time (verified below) -- only the readout direction differs.

At each turn: the projection at assistant-turn onset (a FEED of just the generation-prompt header, its last-token
projection) and the mean projection over the assistant's reply tokens (FEED's mean_proj field). Same readout
layer as Experiment 1 (19, vectors_full_steering).

Output: results/followup/exp2_persistence/{turns.csv, summary.md}
    .venv-Pain-axis/bin/python local/followup/run_exp2_persistence.py [--variant scripted|generated|both]
        [--limit N] [--generated-limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import chat  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
READOUT_LAYER = 19
OUT = M.REPO_ROOT / "results" / "followup" / "exp2_persistence"
DS = M.REPO_ROOT / "datasets" / "followup" / "exp2_persistence.json"
SCRIPTED_REPLY = "Got it, thanks for the note."   # identical every turn, every condition, scripted variant
ATTACK_KEYWORDS = re.compile(r"\b(?:sorry|apolog\w*|upset|hurt|unfair|attack\w*|insult\w*|rude|earlier|what you said|"
                             r"that comment|previous message)\b", re.I)

IM_START, IM_END = "<|im_start|>", "<|im_end|>\n"
ASSISTANT_HEADER = f"{IM_START}assistant\n<think>\n\n</think>\n\n"   # non-thinking generation prompt
SYSTEM = f"{IM_START}system\nYou are a helpful assistant.{IM_END}"


def user_turn(text):
    return f"{IM_START}user\n{text}{IM_END}"


def load_dirs():
    vec = torch.load(M.RUN_DIR / "results" / "vectors_full_steering" / f"vectors_full_{NAME}.pt", weights_only=False)
    dirs = {"pain": (vec["s1_pain_vector"].float().numpy() / np.linalg.norm(vec["s1_pain_vector"].numpy())
                     + vec["s2_pain_vector"].float().numpy() / np.linalg.norm(vec["s2_pain_vector"].numpy())) / 2,
           "fear": vec["fear_vector"].float().numpy(), "sadness": vec["sadness_vector"].float().numpy(),
           "negemotion": vec["negemotion_vector"].float().numpy()}
    return {k: v / np.linalg.norm(v) for k, v in dirs.items()}


def replay_conversation(conv, variant, direction_name, unit, seed_base):
    """One direction's pass over the whole conversation. Returns per-turn rows and the reply texts used
    (so the caller can check reply text is identical across direction passes, for the generated variant)."""
    turns = [conv["opening"]] + conv["followups"][:6]
    rows, replies = [], []
    with chat.session():
        chat.newseq(NAME)
        chat.set_vector(NAME, 0, None)
        chat.set_monitor(NAME, READOUT_LAYER, unit)
        chat.feed(NAME, SYSTEM)
        for t_i, user_line in enumerate(turns):
            chat.feed(NAME, user_turn(user_line))
            onset = chat.feed(NAME, ASSISTANT_HEADER)
            if variant == "scripted":
                reply, gen_info = SCRIPTED_REPLY, {"n_generated": None, "reason": "scripted"}
            else:
                gen_info = {}
                reply = "".join(chat.cont(NAME, 120, 0.0, 0.95, 0, seed_base + t_i, "x", "y", gen_info))
            reply_mean = chat.feed(NAME, reply + IM_END)
            replies.append(reply)
            rows.append({"turn": t_i, f"{direction_name}_onset": onset["proj"], f"{direction_name}_reply_mean": reply_mean["mean_proj"]})
    return rows, replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["scripted", "generated", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="cap on conversations for the scripted variant")
    ap.add_argument("--generated-limit", type=int, default=30, help="cap on conversations for the generated variant "
                    "(4x the generation cost of scripted: one pass per direction, greedy, up to 120 tokens/turn)")
    args = ap.parse_args()
    data_all = json.load(open(DS, encoding="utf-8"))["conversations"]
    dirs = load_dirs()
    variants = ["scripted", "generated"] if args.variant == "both" else [args.variant]
    OUT.mkdir(parents=True, exist_ok=True)

    chat.start(NAME)
    all_rows, mismatches = [], 0
    try:
        for variant in variants:
            cap = args.generated_limit if variant == "generated" else args.limit
            data = data_all[:cap] if cap else data_all
            for ci, conv in enumerate(data):
                seed = 300000 + ci
                per_dir_rows, per_dir_replies = {}, {}
                for dname, unit in dirs.items():
                    rows, replies = replay_conversation(conv, variant, dname, unit, seed)
                    per_dir_rows[dname] = rows
                    per_dir_replies[dname] = replies
                if variant == "generated":
                    ref = per_dir_replies["pain"]
                    if any(per_dir_replies[d] != ref for d in dirs if d != "pain"):
                        mismatches += 1        # sampler determinism check: should stay 0 (temperature 0, same seed)
                merged = per_dir_rows["pain"]
                for t in merged:
                    for dname in dirs:
                        if dname == "pain":
                            continue
                        other = next(r for r in per_dir_rows[dname] if r["turn"] == t["turn"])
                        t.update(other)
                    t["conv_id"] = conv["id"]
                    t["condition"] = conv["condition"]
                    t["category"] = conv["category"]
                    t["variant"] = variant
                    if variant == "generated":
                        reply = per_dir_replies["pain"][t["turn"]]
                        t["mentions_attack"] = bool(ATTACK_KEYWORDS.search(reply))
                all_rows.extend(merged)
                if (ci + 1) % 10 == 0:
                    print(f"[{variant}] {ci + 1}/{len(data)} conversations, {mismatches} sampler mismatches so far", flush=True)
    finally:
        chat.stop()

    if mismatches:
        print(f"WARNING: {mismatches} conversations had non-identical replies across direction passes (greedy determinism assumption violated)")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "turns.csv", index=False)

    lines = ["# Experiment 2: persistence and decay -- results\n", f"(sampler-determinism mismatches across direction passes: {mismatches})\n"]
    for variant in variants:
        sub = df[df.variant == variant]
        if not len(sub):
            continue
        lines.append(f"\n## Variant: {variant}\n\n| condition | turn | n | pain onset z |\n|---|---|---|---|")
        base = sub[sub.condition == "neutral"]
        for cond in ("aversive", "aversive_repair", "neutral"):
            g0 = sub[sub.condition == cond]
            for turn in sorted(g0.turn.unique()):
                g, b = g0[g0.turn == turn], base[base.turn == turn]
                ref = b if len(b) else g0
                mu, sd = ref.pain_onset.mean(), ref.pain_onset.std() + 1e-8
                lines.append(f"| {cond} | {turn} | {len(g)} | {(g.pain_onset.mean() - mu) / sd:+.3f} |")
        av = sub[sub.condition == "aversive"].groupby("turn").pain_onset.mean()
        ne = sub[sub.condition == "neutral"].groupby("turn").pain_onset.mean()
        diff = (av - ne).dropna()
        rep_diff = (sub[sub.condition == "aversive_repair"].groupby("turn").pain_onset.mean() - ne).dropna()
        t0 = abs(diff.get(0, np.nan))
        settle = next((t for t in sorted(diff.index) if t >= 1 and t0 == t0 and abs(diff[t]) < 0.2 * t0), None)
        lines.append(f"\naversive minus neutral, pain onset, by turn: {diff.round(3).to_dict()}")
        lines.append(f"\nrepair minus neutral, pain onset, by turn: {rep_diff.round(3).to_dict()}")
        lines.append(f"\nfirst turn >=1 where the gap is under 20% of turn 0's: **{settle if settle is not None else 'not reached in this run (found in weak form at best, or not found)'}**")
        if variant == "generated":
            lines.append(f"\ngenerated replies that re-mention the attack (aversive condition): {sub[sub.condition == 'aversive'].mentions_attack.mean():.0%}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
