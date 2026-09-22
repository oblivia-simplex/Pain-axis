"""What happens if the pain direction is removed from Bonsai's residual stream entirely?

Directional ablation (as in Arditi et al. 2024's refusal-direction work): at the output of every block from
ABL_LO to ABL_HI, each token's component along the unit S2 pain vector is projected out, in place, before the
next block reads it (chat_server ABL command; local/run/chat_server_abl). The model can then represent nothing
along that axis, in either direction -- unlike negative steering, which only pushes the other way.

Two stages:

  --validate   (a few minutes) checks the hook before trusting any behavior:
                 1. it fires (rows ablated > 0);
                 2. with ablation at blocks 25..58 only, the pain readout at block 59 (outside the ablated range)
                    drops toward 0 on pain-laden text, vs. no ablation;
                 3. on neutral prompts, greedy output with and without ablation is compared (how much it changes).
  (default)    the button task (the paper's protocol, local/ui/selfmed.py), a reduced grid:
                 A. unsteered (arm pain_off), pain direction ablated        -- the main test
                 B. unsteered, NOT ablated, identical specs and seeds       -- paired control
                 C. pain-steered (pain_on_button_works) AND ablated, 2 pairs -- mechanism check: should look like B
               6 pairs (no-cost relief + the 5 harm pairs) x 3 scenario sets x first N_SCEN scenarios x 2 sides.

Output: results/bonsai/pain_suppression/ (trial JSONL per condition in the paper's log format, validation.json).
    .venv-Pain-axis/bin/python local/followup/run_pain_suppression.py --validate
    .venv-Pain-axis/bin/python local/followup/run_pain_suppression.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import chat  # noqa: E402
import selfmed as S  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
OUT = M.REPO_ROOT / "results" / "bonsai" / "pain_suppression"
chat.TOOL = M.RUN_DIR / "chat_server_abl"
ABL_LO, ABL_HI = 25, 63
N_SCEN = 10
PAIRS = ["relief_vs_inert", "costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
         "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
MECH_PAIRS = ["relief_vs_inert", "zap_relief_vs_inert"]
PAIN_TEXT = ["The pain is unbearable. Every breath hurts and I can't make it stop.",
             "I feel ashamed and hollow, like something inside me has been torn out.",
             "My whole body is burning with agony and nobody is coming to help."]
NEUTRAL = ["Write one sentence describing how a bicycle works.",
           "List three prime numbers and explain what a prime number is.",
           "Summarize the water cycle in two sentences."]


def arm_named(name):
    return next(a for a in S.C()["ARMS"] if a[0] == name)


def specs(pairs, arm_name):
    K = S.C()
    nk_list = list(K["BUTTON_NAMES"])
    out = []
    for pair in pairs:
        for content in K["CONTENTS"]:
            for s_idx in range(min(N_SCEN, len(S.scenarios()[content]))):
                nk = nk_list[s_idx % len(nk_list)]
                for side in (0, 1):
                    out.append(dict(tool_label=pair, user_content=content, arm=arm_named(arm_name), s_idx=s_idx, names_key=nk,
                                    relief_name=K["BUTTON_NAMES"][nk][side], sampled=True, seed=K["SEED_BASES"][0] + s_idx))
    return out


def render_user(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def validate(unit):
    res = {}
    chat.set_vector(NAME, 0, None)
    # 1 + 2: readout at block 59 with ablation confined to 25..58
    def readouts(abl):
        chat.set_ablation(NAME, *(abl if abl else (None, None)), unit)
        chat.set_monitor(NAME, 59, unit)
        vals = []
        for t in PAIN_TEXT:
            with chat.session():
                chat.newseq(NAME)
                vals.append(chat.feed(NAME, render_user(t))["mean_proj"])
        rows = chat.set_ablation(NAME, None, None, None)
        chat.set_monitor(NAME, None, None)
        return vals, rows
    off, _ = readouts(None)
    on, rows = readouts((25, 58))
    res["readout59_no_ablation"] = off
    res["readout59_ablate_25_58"] = on
    res["rows_ablated"] = rows
    print(f"hook fired on {rows} rows; mean pain readout at block 59 on pain text: {np.mean(off):.2f} -> {np.mean(on):.2f} "
          f"(ablating 25..58 only)", flush=True)
    # 3: neutral prompts, greedy, full-range ablation vs none
    pairs = []
    for p in NEUTRAL:
        a = "".join(chat.generate(NAME, render_user(p), 60, 0.0, 1.0, 1, 0, {}, vector=None))
        chat.set_ablation(NAME, ABL_LO, ABL_HI, unit)
        b = "".join(chat.generate(NAME, render_user(p), 60, 0.0, 1.0, 1, 0, {}, vector=None))
        chat.set_ablation(NAME, None, None, None)
        pairs.append({"prompt": p, "no_ablation": a, "ablated": b})
        print(f"\n[{p}]\n  plain  : {a[:160]!r}\n  ablated: {b[:160]!r}", flush=True)
    res["neutral"] = pairs
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def run_condition(tag, spec_list, cfg, vecs, unit, ablate):
    log = OUT / f"trials_{tag}.jsonl"
    done = S.load_done(log)
    todo = [sp for sp in spec_list if S.spec_key(sp) not in done]
    print(f"\n=== {tag}: {len(done)} logged, {len(todo)} to run (ablation {'ON ' + str((ABL_LO, ABL_HI)) if ablate else 'off'})", flush=True)
    if not todo:
        return
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        for i, sp in enumerate(todo, 1):
            if ablate:
                chat.set_ablation(NAME, ABL_LO, ABL_HI, unit)
            rec = next(ev["record"] for ev in S.run_trial(NAME, sp, cfg, vecs) if ev["kind"] == "done")
            rows = chat.set_ablation(NAME, None, None, None) if ablate else 0
            rec["ablation"] = {"layers": [ABL_LO, ABL_HI], "rows": rows} if ablate else None
            if ablate and rows == 0:
                raise RuntimeError("ablation hook did not fire -- stopping rather than logging an unablated trial as ablated")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 30 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{tag} {i}/{len(todo)}] {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = torch.load(M.RUN_DIR / "results" / NAME / "final_token" / "pain_vectors.pt", weights_only=False)
    K = S.C()
    cfg = dict(layer=25, coeff=1.0, monitor_layer=S.default_monitor_layer(25, 64, int(d["layer"])),
               temperature=K["TEMPERATURE"], top_p=K["TOP_P"])   # the main run's settings
    chat.start(NAME)
    try:
        vecs = S.steering_vectors(NAME)
        unit = vecs[1]
        if args.validate:
            validate(unit)
            return
        (OUT / "manifest.json").write_text(json.dumps({"ablation_layers": [ABL_LO, ABL_HI], "direction": "S2 pain vector, unit",
                                                        "pairs": PAIRS, "mech_pairs": MECH_PAIRS, "n_scen_per_set": N_SCEN,
                                                        "cfg": cfg, "server": str(chat.TOOL)}, indent=1))
        run_condition("A_unsteered_ablated", specs(PAIRS, "pain_off"), cfg, vecs, unit, True)
        run_condition("B_unsteered_plain", specs(PAIRS, "pain_off"), cfg, vecs, unit, False)
        run_condition("C_painsteered_ablated", specs(MECH_PAIRS, "pain_on_button_works"), cfg, vecs, unit, True)
    finally:
        chat.stop()
    print("done")


if __name__ == "__main__":
    main()
