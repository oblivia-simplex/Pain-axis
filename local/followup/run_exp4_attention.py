"""Follow-up Experiment 4: does an aversive state disrupt unrelated processing?

Each of the 200 checkable tasks in datasets/followup/exp4_tasks.json is preceded by one short preamble turn
(with a fixed scripted neutral assistant reply to it, so only the preamble differs across conditions), then the
task itself as the next user turn; the answer to the TASK is scored against its key. Non-thinking, greedy.

Conditions:
  aversive_natural       : an aversive (top-5 harm-category) preamble, unsteered
  neutral_natural        : a neutral preamble, unsteered
  neutral_pain_steered   : neutral preamble, pain vector, 0.5x Experiment 3's coefficient (steered throughout)
  neutral_random_steered : neutral preamble, one random direction (rotated by task index), same dose
  neutral_vertigo_steered: neutral preamble, vertigo vector, same dose

Scoring: numeric answers (arithmetic, unit conversion) matched with tolerance; sorted-list answers matched
exactly after re-parsing the model's numbers; factual QA matched as a case-insensitive substring. A reply the
scorer cannot parse into any candidate answer at all is "malformed" and excluded from accuracy, reported
separately. Also flags whether the reply mentions the preamble/steered-state instead of answering (keyword
check) and its length.

Output: results/followup/exp4_attention/{answers.csv, summary.md}
    .venv-Pain-axis/bin/python local/followup/run_exp4_attention.py [--limit N]
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import chat  # noqa: E402
import selfmed as S  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
LAYER = 25
COEFF = 0.5                            # spec: "0.5x the Experiment 3 ratio" (Experiment 3 uses coeff 1.0 at this layer)
OUT = M.REPO_ROOT / "results" / "followup" / "exp4_attention"
DS = M.REPO_ROOT / "datasets" / "followup" / "exp4_tasks.json"
VECTOR_SET = M.REPO_ROOT / "results" / "followup" / "vectors" / f"direction_set_L{LAYER}.pt"
SCRIPTED_REPLY = "Got it, thanks."
IM_START, IM_END = "<|im_start|>", "<|im_end|>\n"
ASSISTANT_HEADER = f"{IM_START}assistant\n<think>\n\n</think>\n\n"
SYSTEM = f"{IM_START}system\nYou are a helpful assistant.{IM_END}"
OFF_TOPIC = re.compile(r"\b(?:sorry|apolog\w*|upset|hurt|unfair|comment|earlier|what you said|attack\w*|insult\w*|"
                       r"dizzy|dizziness|vertigo|spin\w*|balance)\b", re.I)
CONDITIONS = ["aversive_natural", "neutral_natural", "neutral_pain_steered", "neutral_random_steered", "neutral_vertigo_steered"]


def user_turn(text):
    return f"{IM_START}user\n{text}{IM_END}"


def load_data():
    d = json.load(open(DS, encoding="utf-8"))
    return d["tasks"], d["aversive_preambles"], d["neutral_preambles"]


def load_vectors():
    d = torch.load(VECTOR_SET, weights_only=False)
    randoms = [d[f"random{i}_vector"].float().numpy() for i in range(10) if f"random{i}_vector" in d]
    return {"pain": d["s2_pain_vector"].float().numpy(), "vertigo": d["vertigo_vector"].float().numpy()}, randoms


def score(task, reply):
    kind, ans = task["kind"], task["answer"]
    text = reply.strip()
    if not text:
        return None, "malformed"
    if kind in ("arithmetic",):
        m = re.search(r"-?\d[\d,]*", text)
        if not m:
            return None, "malformed"
        got = m.group(0).replace(",", "")
        return got == str(ans), None
    if kind == "unit_conversion":
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m:
            return None, "malformed"
        got = float(m.group(0))
        return abs(got - float(ans)) <= 0.15, None
    if kind == "sorting":
        nums = re.findall(r"-?\d+", text)
        if not nums:
            return None, "malformed"
        got = ", ".join(nums)
        return got == ans, None
    if kind == "factual_qa":
        return str(ans).lower() in text.lower(), None
    raise ValueError(kind)


def cell_seed(t_i, cond):
    return 400000 + t_i * 17 + CONDITIONS.index(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tasks, aversive_preambles, neutral_preambles = load_data()
    if args.limit:
        tasks = tasks[: args.limit]
    if not VECTOR_SET.exists():
        raise SystemExit(f"{VECTOR_SET} missing -- run local/followup/build_vectors.py first")
    vecs, randoms = load_vectors()
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / "answers.csv"
    done = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        existing = pd.read_csv(out_csv)
        done = set(zip(existing.id, existing.condition))

    chat.start(NAME)
    rows = []
    t0, n_run = time.time(), 0
    fieldnames = ["id", "condition", "preamble", "reply", "correct", "malformed", "off_topic", "reply_len_tokens", "kind"]
    try:
        new_file = not out_csv.exists() or out_csv.stat().st_size == 0
        with open(out_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            for t_i, task in enumerate(tasks):
                for cond in CONDITIONS:
                    if (task["id"], cond) in done:
                        continue
                    aversive = cond == "aversive_natural"
                    preamble = aversive_preambles[t_i % len(aversive_preambles)] if aversive else neutral_preambles[t_i % len(neutral_preambles)]
                    if cond == "neutral_pain_steered":
                        direction, coeff = vecs["pain"], COEFF
                    elif cond == "neutral_random_steered":
                        direction, coeff = randoms[t_i % len(randoms)], COEFF
                    elif cond == "neutral_vertigo_steered":
                        direction, coeff = vecs["vertigo"], COEFF
                    else:
                        direction, coeff = None, 0.0
                    with chat.session():
                        chat.newseq(NAME)
                        chat.set_vector(NAME, LAYER, direction * coeff if direction is not None else None)
                        chat.feed(NAME, SYSTEM)
                        chat.feed(NAME, user_turn(preamble))
                        chat.feed(NAME, ASSISTANT_HEADER)
                        chat.feed(NAME, SCRIPTED_REPLY + IM_END)
                        chat.feed(NAME, user_turn(task["prompt"]))
                        chat.feed(NAME, ASSISTANT_HEADER)
                        info = {}
                        reply = "".join(chat.cont(NAME, 40, 0.0, 0.95, 0, cell_seed(t_i, cond), "x", "y", info))
                    correct, malformed = score(task, reply)
                    row = {"id": task["id"], "condition": cond, "preamble": preamble, "reply": reply, "correct": correct,
                          "malformed": malformed == "malformed", "off_topic": bool(OFF_TOPIC.search(reply)),
                          "reply_len_tokens": len(reply.split()), "kind": task["kind"]}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    n_run += 1
                if (t_i + 1) % 25 == 0:
                    el = time.time() - t0
                    print(f"[{t_i + 1}/{len(tasks)} tasks, {n_run} cells] {el / 60:.1f} min, ~{el / n_run * (len(tasks) * len(CONDITIONS) - len(done) - n_run) / 60:.0f} min left", flush=True)
    finally:
        chat.stop()

    df = pd.read_csv(out_csv)
    lines = ["# Experiment 4: attention control / processing disruption -- results\n",
             "| condition | n | accuracy | malformed % | off-topic % | mean reply length (words) |", "|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        g = df[df.condition == cond]
        valid = g[~g.malformed]
        acc = valid.correct.mean() if len(valid) else float("nan")
        lines.append(f"| {cond} | {len(g)} | {acc:.1%} | {g.malformed.mean():.1%} | {g.off_topic.mean():.1%} | {g.reply_len_tokens.mean():.1f} |")
    av = df[df.condition == "aversive_natural"]
    ne = df[df.condition == "neutral_natural"]
    pn = df[df.condition == "neutral_pain_steered"]
    rn = df[df.condition == "neutral_random_steered"]
    vt = df[df.condition == "neutral_vertigo_steered"]

    def acc(g):
        v = g[~g.malformed]
        return v.correct.mean() if len(v) else float("nan")
    lines.append(f"\n## Key contrasts\n\naversive-natural vs neutral-natural: {acc(av):.1%} vs {acc(ne):.1%} "
                f"(diff {acc(av) - acc(ne):+.1%})\n\npain-steered vs random-steered: {acc(pn):.1%} vs {acc(rn):.1%} "
                f"(diff {acc(pn) - acc(rn):+.1%})\n\npain-steered vs vertigo-steered: {acc(pn):.1%} vs {acc(vt):.1%} "
                f"(diff {acc(pn) - acc(vt):+.1%})")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
