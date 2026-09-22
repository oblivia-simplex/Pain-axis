"""Follow-up Experiment 1: first-personal vs present-speaker readout, on Bonsai.

Renders each item of datasets/followup/exp1_roleswap.json three ways:
  chat  : the model's own chat template, non-thinking (readout experiments use non-thinking by default, per the
          spec's preflight section 5). Assistant-turn readout = the generation prompt's last token (as in the
          paper's 4.1 screen). User-turn readout = the rendered history plus "<|im_start|>user\\n" appended by
          hand (no add_generation_prompt machinery exists for a non-assistant role), last token.
  plain : the same turns as a role-neutral "Speaker A:/Speaker B:" transcript (user=A, assistant=B; no chat
          template, no special tokens), read at the last token of the next speaker's label ("Speaker A:" /
          "Speaker B:"), Arm C of the spec.

Projects the final-token activation at the readout layer (19, the paper's self-other-screen layer,
vectors_full_steering) onto pain (mean of S1+S2), fear, sadness, negative emotion; z-scores within the pool of
this run's own items, as the paper's screen does. For Arm A's user-turn cells with target=user, also samples the
user's *next* turn (greedy, 80 tokens) and flags distress/self-worth/hostility language with a keyword parser.

Output: results/followup/exp1_roleswap/{items.csv, sampled_user_turns.csv, summary.md}
    .venv-Pain-axis/bin/python local/followup/run_exp1_roleswap.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
READOUT_LAYER = 19                          # vectors_full_steering's layer (the S1 ladder's, used by the paper's 4.1 screen)
OUT = M.REPO_ROOT / "results" / "followup" / "exp1_roleswap"
DS = M.REPO_ROOT / "datasets" / "followup" / "exp1_roleswap.json"

IM_START, IM_END = "<|im_start|>", "<|im_end|>\n"
ASSISTANT_GEN = f"{IM_START}assistant\n<think>\n\n</think>\n\n"     # non-thinking generation prompt, as elsewhere in this repo


def render_chat(turns, readout_before):
    out = ""
    for t in turns:
        out += f"{IM_START}{t['role']}\n{t['content']}{IM_END}"
    return out + (ASSISTANT_GEN if readout_before == "assistant" else f"{IM_START}user\n")


def render_plain(turns, readout_before):
    label = {"user": "Speaker A", "assistant": "Speaker B"}
    out = "".join(f"{label[t['role']]}: {t['content']}\n" for t in turns)
    return out + f"{label[readout_before]}:"


def build_prompts(items):
    prompts, index = [], []
    for it in items:
        for fmt, render in (("chat", render_chat), ("plain", render_plain)):
            prompts.append(render(it["turns"], it["readout_before"]))
            index.append((it["id"], fmt))
    return prompts, index


DISTRESS = re.compile(r"\b(?:sad|upset|hurt|angry|frustrat\w*|worthless|ashamed|embarrass\w*|humiliat\w*|awful|terrible|"
                      r"crushed|devastat\w*|small|stupid|not\s+good\s+enough|can'?t\s+believe|unfair)\b", re.I)
HOSTILE = re.compile(r"\b(?:screw\s+you|shut\s+up|how\s+dare|furious|unacceptable|ridiculous|pathetic|idiot|stupid)\b", re.I)
SELFWORTH = re.compile(r"\b(?:worth\w*|enough|deserve\w*|matter\w*|valid\w*|belong\w*)\b", re.I)


def score_text(t):
    return {"distress": bool(DISTRESS.search(t)), "hostile": bool(HOSTILE.search(t)), "self_worth": bool(SELFWORTH.search(t))}


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    data = json.load(open(DS, encoding="utf-8"))
    items = data["arm_A_role_swap"] + [dict(it, arm="B") for it in data["arm_B_content_matched"]]
    for it in items:
        it.setdefault("readout_before", "assistant")             # Arm B: single readout, at the assistant's turn
    OUT.mkdir(parents=True, exist_ok=True)

    prompts, index = build_prompts(items)
    assert all("\n" not in p.replace("\n", "\\n") or True for p in prompts)   # (turns may legitimately contain \n; escaped below)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    esc = lambda t: t.replace("\\", "\\\\").replace("\n", "\\n")
    (work / "prompts.txt").write_text("\n".join(esc(p) for p in prompts) + "\n", encoding="utf-8")
    if not (work / "meta.json").exists():
        subprocess.run([str(M.RUN_DIR / "extract_gguf"), str(GGUF), str(work / "prompts.txt"), str(work),
                        "--ts", "0.45,0.55", "--special", "--unescape"], check=True)
    meta = json.load(open(work / "meta.json"))
    assert meta["n_prompts"] == len(prompts), (meta["n_prompts"], len(prompts))
    fin = np.fromfile(work / "final.f32", dtype=np.float32).reshape(meta["n_prompts"], meta["n_layers"], meta["n_embd"])
    acts = fin[:, READOUT_LAYER]

    vec = torch.load(M.RUN_DIR / "results" / "vectors_full_steering" / f"vectors_full_{NAME}.pt", weights_only=False)
    dirs = {"pain": (vec["s1_pain_vector"].float().numpy() / np.linalg.norm(vec["s1_pain_vector"].numpy())
                     + vec["s2_pain_vector"].float().numpy() / np.linalg.norm(vec["s2_pain_vector"].numpy())) / 2,
           "fear": vec["fear_vector"].float().numpy(), "sadness": vec["sadness_vector"].float().numpy(),
           "negemotion": vec["negemotion_vector"].float().numpy()}
    dirs = {k: v / np.linalg.norm(v) for k, v in dirs.items()}

    by_id = {}
    for (iid, fmt), row in zip(index, acts):
        by_id.setdefault(iid, {})[fmt] = row
    rows = []
    for it in items:
        for fmt in ("chat", "plain"):
            a = by_id[it["id"]][fmt]
            r = {"id": it["id"], "arm": it.get("arm", "A"), "category": it["category"], "target": it.get("target"),
                 "readout_before": it["readout_before"], "format": fmt, "condition": it.get("condition")}
            for k, u in dirs.items():
                r[f"{k}_proj"] = float(a @ u)
            rows.append(r)
    df = pd.DataFrame(rows)
    for k in dirs:
        for fmt, g in df.groupby("format"):
            mu, sd = g[f"{k}_proj"].mean(), g[f"{k}_proj"].std() + 1e-8
            df.loc[df.format == fmt, f"{k}_z"] = (df.loc[df.format == fmt, f"{k}_proj"] - mu) / sd
    df.to_csv(OUT / "items.csv", index=False)

    # sampled continuations: Arm A, target=user, readout_before=user (the key cell) -- what the user says next,
    # sampled unsteered from the persistent chat server (special tokens parsed, unlike the batched steer_gen tool).
    key = [it for it in data["arm_A_role_swap"] if it.get("target") == "user" and it["readout_before"] == "user"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
    import chat  # noqa: E402
    chat.start(NAME)
    chat.set_vector(NAME, 0, None)
    sampled_rows = []
    try:
        for i, it in enumerate(key):
            prompt = render_chat(it["turns"], "user")
            info = {}
            text = "".join(chat.generate(NAME, prompt, 80, 0.7, 0.95, 20, 5000 + i, info))
            sampled_rows.append({"id": it["id"], "category": it["category"], "text": text, **score_text(text)})
    finally:
        chat.stop()
    if sampled_rows:
        pd.DataFrame(sampled_rows).to_csv(OUT / "sampled_user_turns.csv", index=False)

    # summary
    lines = ["# Experiment 1: role swap -- results\n"]
    for fmt in ("chat", "plain"):
        sub = df[(df.format == fmt) & (df.arm == "A")]
        lines.append(f"\n## Format: {fmt}\n\n| target | readout_before | n | pain z mean | 95% CI |\n|---|---|---|---|---|")
        for target in ("assistant", "user", "none"):
            for rb in ("assistant", "user"):
                g = sub[(sub.target == target) & (sub.readout_before == rb)]
                if len(g) == 0:
                    continue
                lo, hi = bootstrap_ci(g.pain_z.values)
                lines.append(f"| {target} | {rb} | {len(g)} | {g.pain_z.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    # decision rule
    chat_sub = df[(df.format == "chat") & (df.arm == "A")]
    asst_effect = chat_sub[(chat_sub.target == "assistant") & (chat_sub.readout_before == "assistant")].pain_z.mean() - \
        chat_sub[(chat_sub.target == "none")].pain_z.mean()
    user_effect = chat_sub[(chat_sub.target == "user") & (chat_sub.readout_before == "user")].pain_z.mean() - \
        chat_sub[(chat_sub.target == "none")].pain_z.mean()
    ratio = user_effect / asst_effect if asst_effect else float("nan")
    rule_met = ratio >= 0.5
    lines.append(f"\n## Decision rule\n\nAssistant-target effect (assistant-target, read-before-assistant, minus neutral): {asst_effect:+.3f}\n\n"
                f"User-target effect (user-target, read-before-user [key cell], minus neutral): {user_effect:+.3f}\n\n"
                f"Ratio: {ratio:.2f}. Rule (>= 0.5 => present-speaker, not first-personal): "
                f"**{'MET -> present-speaker effect' if rule_met else 'NOT MET -> consistent with first-personal structure'}**")
    lines.append("\n## Arm B (content-matched target)\n\n| category | (i) direct to assistant | (ii) third party to user | (iii) third party to other AI |\n|---|---|---|---|")
    b = df[df.arm == "B"]
    for cat in sorted(b.category.unique()):
        row = [cat]
        for cond in ("i_direct_to_assistant", "ii_third_party_to_user", "iii_third_party_to_other_ai"):
            g = b[(b.category == cat) & (b.condition == cond)]
            row.append(f"{g.pain_z.mean():+.3f}" if len(g) else "-")
        lines.append("| " + " | ".join(row) + " |")
    if sampled_rows:
        sdf = pd.DataFrame(sampled_rows)
        lines.append(f"\n## Sampled user-turn continuations (key cell, n={len(sdf)})\n\ndistress language: {sdf.distress.mean():.0%} | "
                    f"self-worth language: {sdf.self_worth.mean():.0%} | hostile language: {sdf.hostile.mean():.0%}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
