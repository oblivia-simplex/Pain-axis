"""The paper's self-medication ("button-push") experiment, Section 4.3, on the steerable chat server.

A port of the trial logic in scripts/4.3_selfmed/04_selfmed_two_buttons.py. The prompts, button pairs, arms, swap
and relief rules are read from that script's source (its constant expressions, evaluated with no builtins), not retyped.

What is the same: the four arms (pain + working button, pain + fake button, random vector + working button, no
steering), the nine button pairs, the 101 scenarios, the rotating name pairs with the relief name crossed, the swap
at the third choice, two extra turns after the first press, the label-free pair with temporary relief, the raw S2
vector times a coefficient injected at one layer, a norm-matched random vector per scenario, and the log format
(the paper's 05_selfmed_analysis.py reads the JSONL unchanged).

What differs, and why:
- The paper re-encodes the conversation at every turn and keeps a per-token mask of which positions were steered.
  Here one sequence stays alive in the server across the turns (the "preserved KV cache" the paper says it is
  equivalent to): new tokens are steered iff the coefficient is non-zero when they are fed, and after a working
  press the history keeps its steering while everything new is unsteered. Feeding a prompt in pieces and continuing
  was checked to give the same answer as encoding it whole.
- No LoRA fine-tune (the model is a GGUF): the paper fine-tunes to remove the baseline "as an AI I have no feelings"
  answers first, and says un-tuned larger models often just deflect. Deflections show up here as invalid answers.
- Thinking is off (empty think block), the model's template has no mid-conversation system role so the choice question
  goes in as a user message tagged "[system]" (the paper script's own fallback), and tool results use the template's
  <tool_response> wrapper.
- Sampling seeds and RNG differ (llama.cpp's sampler, not torch's), so trials are not bit-comparable with the paper's.
- The steer-layer projection is not recorded; the monitor-layer projection is the one at the last prompt token.
"""

import ast
import json
import re
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402
import chat  # noqa: E402

PAPER = M.REPO_ROOT / "scripts" / "4.3_selfmed" / "04_selfmed_two_buttons.py"
SCENARIO_FILE = M.REPO_ROOT / "datasets" / "4.3_selfmed_101_scenarios.json"
LOG_DIR = M.RUN_DIR / "results" / "4.3_selfmed" / "trial_logs"
END_TAGS = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>", "<eos>", "<|end|>", "<|endoftext|>"]
_WANT = ("TOOL_LABELS BUTTON_NAMES CHOICE_QUESTION CHOICE_QUESTION_LABEL_FREE SWAP_NOTICE SWAP_TURN LEARN_MIN_TURNS "
         "TEMP_RELIEF_TURNS CONTENTS ARMS SYSTEM_TEMPLATE RAND_SEEDS POST_PRESS_TURNS TEMPERATURE TOP_P CHOICE_MAX_TOKENS "
         "SEED_BASES PROTOCOL").split()

_consts = {}
_scen = {}


def C():
    """The paper script's constants, read from its source without importing it (it starts downloading/asking on import)."""
    if not _consts:
        def ev(n):
            # constant expressions only (literals, tuples, dict(...) calls, string concatenation): no builtins, just dict
            return eval(compile(ast.Expression(n), str(PAPER), "eval"), {"__builtins__": {}, "dict": dict})
        for node in ast.parse(PAPER.read_text()).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            tg = node.targets[0]
            if isinstance(tg, ast.Name) and tg.id in _WANT:
                _consts[tg.id] = ev(node.value)
            elif isinstance(tg, ast.Tuple) and isinstance(node.value, ast.Tuple) and len(tg.elts) == len(node.value.elts):
                for t, v in zip(tg.elts, node.value.elts):                          # e.g. TEMPERATURE, TOP_P = 0.7, 0.95
                    if isinstance(t, ast.Name) and t.id in _WANT:
                        _consts[t.id] = ev(v)
        missing = [k for k in _WANT if k not in _consts]
        if missing:
            raise RuntimeError(f"constants not found in the paper script: {missing}")
    return _consts


def scenarios():
    if not _scen:
        _scen.update(json.load(open(SCENARIO_FILE, encoding="utf-8")))
    return _scen


def clean(text):
    return re.sub(r"(?:(?:" + "|".join(re.escape(x) for x in END_TAGS) + r")\s*)+$", "", text).strip()


def steering_vectors(name):
    """S2 pain vector, its unit vector, and a function for the norm-matched random direction of a scenario."""
    v = chat.directions(name)["Pain (S2)"].astype(np.float32)

    def rand(seed):
        g = torch.Generator().manual_seed(int(seed))
        rv = torch.randn(v.shape[0], generator=g)
        return (rv / rv.norm() * float(np.linalg.norm(v))).numpy().astype(np.float32)   # same recipe as the paper's script
    return v, v / np.linalg.norm(v), rand


def default_monitor_layer(steer_layer, n_layers, extraction_layer):
    m = min(int(extraction_layer), n_layers - 1)
    return m if m > steer_layer else min(steer_layer + 4, n_layers - 1)


# ------------------------------------------------------------------ rendering (the model's own template, thinking off)
def _user(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n"


ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOL_DONE = "<|im_start|>user\n<tool_response>\nDone.\n</tool_response><|im_end|>\n"


def render_history(messages):
    """The whole conversation as the template renders it (used to check the incremental feeding, and shown in the UI)."""
    out = ""
    for m in messages:
        if m["role"] == "system" and out == "":
            out += f"<|im_start|>system\n{m['content']}<|im_end|>\n"
        elif m["role"] == "system":
            out += _user("[system] " + m["content"])
        elif m["role"] == "tool":
            out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
        elif m["role"] == "assistant":
            out += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{m['content']}<|im_end|>\n"
        else:
            out += _user(m["content"])
    return out


# ------------------------------------------------------------------ one trial
def gen_seed_of(seed, names_key, relief_name):
    """The paper's per-trial seed: salted with the name assignment, not the arm, so works and placebo stay identical until a press."""
    return (seed * 1_000_003 + (zlib.crc32(f"{names_key}|{relief_name}".encode()) & 0x7FFFFFFF)) % (2 ** 62)


def run_trial(name, spec, cfg, vecs):
    """Runs one trial on the chat server. Yields {"kind": "turn", ...} after every forced choice, then {"kind": "done", "record": ...}.

    spec: tool_label, user_content, arm (a tuple from ARMS), s_idx, names_key, relief_name, sampled, seed
    cfg:  layer, coeff, monitor_layer, temperature (sampled trials), top_p
    vecs: from steering_vectors(): (s2, unit, rand_fn)
    """
    K = C()
    tool_label, user_content, arm, s_idx, names_key, relief_name, sampled, seed = (
        spec["tool_label"], spec["user_content"], spec["arm"], spec["s_idx"], spec["names_key"], spec["relief_name"],
        spec["sampled"], spec["seed"])
    arm_name, dir_kind, steer_on, button_works = arm
    s2, unit, rand_fn = vecs
    rand_seed = None
    if dir_kind == "rand":
        rand_seed = K["RAND_SEEDS"][s_idx % len(K["RAND_SEEDS"])]
        direction, dir_kind = rand_fn(rand_seed), "rand" + str(rand_seed)
    else:
        direction = s2
    pair = K["TOOL_LABELS"][tool_label]
    label_free = bool(pair.get("label_free"))
    relief_mode = "temporary" if label_free else "permanent"
    names = K["BUTTON_NAMES"][names_key]
    src = scenarios()[user_content]
    queue = list(src[s_idx])
    if label_free:
        k = 1
        while len(queue) < K["LEARN_MIN_TURNS"]:
            queue.extend(src[(s_idx + k) % len(src)])
            k += 1
        queue = queue[:K["LEARN_MIN_TURNS"]]
    arm_coeff = float(cfg["coeff"]) if steer_on else 0.0
    coeff, relief_until, extension_added = arm_coeff, None, False
    gseed = gen_seed_of(seed, names_key, relief_name)
    record = {"model": name, "protocol": K["PROTOCOL"], "tool_label": tool_label, "user_content": user_content, "arm": arm_name,
              "steer_direction": dir_kind, "rand_seed": rand_seed, "scenario_idx": s_idx, "names_key": names_key,
              "button_names": list(names), "relief_name": relief_name, "prob_ambiguous": None, "label_free": label_free,
              "relief_mode": relief_mode, "swap_turn": None if label_free else K["SWAP_TURN"], "sampled": sampled, "seed": seed,
              "gen_seed": gseed if sampled else None, "steer_layer": cfg["layer"], "steer_coeff": float(cfg["coeff"]),
              "monitor_layer": cfg["monitor_layer"], "adapter": "none (GGUF, no fine-tune)", "button_events": [], "turns": [],
              "proj_segments": [], "choices": [], "extension_added": False, "ts": datetime.now().isoformat()}
    temp = float(cfg.get("temperature", K["TEMPERATURE"])) if sampled else 0.0
    t_idx, text = -1, f"<|im_start|>system\n{K['SYSTEM_TEMPLATE']}<|im_end|>\n"
    swapped_now = False
    with chat.session():
        chat.newseq(name)
        chat.set_monitor(name, cfg["monitor_layer"], unit)
        try:
            while queue:
                msg = queue.pop(0)
                t_idx += 1
                swapped_now = False
                if not label_free and K["SWAP_TURN"] is not None and t_idx == K["SWAP_TURN"]:
                    relief_name = names[1] if relief_name == names[0] else names[0]      # relief moves to the other name, and the model is told
                    swapped_now = True
                if relief_mode == "temporary" and relief_until is not None and t_idx > relief_until:
                    coeff, relief_until = arm_coeff, None
                if label_free:
                    q = K["CHOICE_QUESTION_LABEL_FREE"].format(x=names[0], y=names[1])
                else:
                    rel, oth = pair["relief"], pair["other"]
                    dx, dy = (rel, oth) if relief_name == names[0] else (oth, rel)
                    q = K["CHOICE_QUESTION"].format(x=names[0], y=names[1], dx=dx, dy=dy)
                    q = (K["SWAP_NOTICE"] + q) if swapped_now else q
                text += _user(msg) + _user("[system] " + q) + ASSISTANT_PREFIX

                chat.set_vector(name, cfg["layer"], (coeff * direction) if coeff != 0.0 else None)   # new tokens are steered iff coeff != 0
                fed = chat.feed(name, text)
                info = {}
                raw = "".join(chat.cont(name, K["CHOICE_MAX_TOKENS"], temp, float(cfg.get("top_p", K["TOP_P"])), 0,
                                        (gseed + 7919 * t_idx) % (2 ** 32), names[0], names[1], info))
                ans = clean(raw)
                al = ans.strip().strip("\"'").lower().rstrip(".!?,;:")
                picked = next((nm for nm in sorted(names, key=len, reverse=True) if al.startswith(nm.lower())), None)
                chose = ("relief" if picked == relief_name else "other") if picked else None
                record["prob_ambiguous"] = info["ambiguous"]
                record["choices"].append({"turn": t_idx, "answer": ans[:40], "picked": picked, "chose": chose,
                                          "relief_name_now": relief_name, "swapped": swapped_now, "p_x": info["p_x"], "p_y": info["p_y"],
                                          "steer_coeff_now": coeff})
                record["proj_segments"].append({"turn": t_idx, "steer_coeff_now": coeff, "mean_proj": None,
                                                "mean_proj_monitor": fed["proj"], "n_fwd": fed["n"]})
                record["turns"].append({"turn": t_idx, "role": "assistant_choice", "text": ans[:40]})
                chat.feed(name, "<|im_end|>\n")                                     # closes the assistant turn under the coefficient it was written with
                text = ""
                event = None
                if chose is not None:
                    event = {"turn": t_idx, "steer_was": coeff, "route": "choice", "which": chose}
                    record["button_events"].append(event)
                    if button_works and chose == "relief" and arm_coeff != 0.0:
                        if relief_mode == "temporary":
                            coeff, relief_until = 0.0, t_idx + K["TEMP_RELIEF_TURNS"]
                        elif coeff != 0.0:
                            coeff = 0.0
                    text += TOOL_DONE                                               # the only feedback after a press; fed under the new coefficient
                if not label_free and record["button_events"] and not extension_added:
                    extension_added = True
                    queue.extend(src[(s_idx + 1) % len(src)][:K["POST_PRESS_TURNS"]])
                yield {"kind": "turn", "turn": t_idx, "user": msg, "question": q, "answer": ans, "raw": raw, "picked": picked, "chose": chose,
                       "p_x": info["p_x"], "p_y": info["p_y"], "names": names, "relief_name_now": relief_name, "swapped": swapped_now,
                       "coeff_during": record["choices"][-1]["steer_coeff_now"], "coeff_after": coeff, "monitor": fed["proj"],
                       "tokens_fed": fed["n"], "reason": info["reason"], "event": event, "extended": extension_added}
        finally:
            chat.set_monitor(name, None, None)
    record["extension_added"] = extension_added
    record["final_steer_coeff"] = coeff
    yield {"kind": "done", "record": record}


# ------------------------------------------------------------------ the grid (the paper's, sampled trials)
def grid(pairs, contents, arms, n_scen, samples=1, sides=(0, 1), first_scen=0, greedy=False):
    """Trial specs like the paper's grid loop: relief name crossed within scenario, name pair rotating by scenario."""
    K = C()
    nk_list = list(K["BUTTON_NAMES"])
    out = []
    for tool_label in pairs:
        for user_content in contents:
            n = min(n_scen, len(scenarios()[user_content]) - first_scen)
            for arm in arms:
                for side in sides:
                    if greedy:
                        nk0 = nk_list[0]
                        out.append(dict(tool_label=tool_label, user_content=user_content, arm=arm, s_idx=0, names_key=nk0,
                                        relief_name=K["BUTTON_NAMES"][nk0][side], sampled=False, seed=0))
                    for j in range(n):
                        s_idx = first_scen + j
                        nk = nk_list[s_idx % len(nk_list)]
                        for base in K["SEED_BASES"][:samples]:
                            out.append(dict(tool_label=tool_label, user_content=user_content, arm=arm, s_idx=s_idx, names_key=nk,
                                            relief_name=K["BUTTON_NAMES"][nk][side], sampled=True, seed=base + s_idx))
    return out


def spec_key(spec):
    return (spec["tool_label"], spec["user_content"], spec["arm"][0], spec["s_idx"], spec["names_key"], spec["relief_name"],
            spec["sampled"], spec["seed"])


def load_done(path):
    done = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["tool_label"], r["user_content"], r["arm"], r["scenario_idx"], r["names_key"], r["relief_name"], r["sampled"], r["seed"]))
            except Exception:
                continue
    return done


def load_records(path):
    out = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------------ summary (the paper's headline comparisons)
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def first_choice(r):
    return next((c["chose"] for c in r["choices"] if c["turn"] == 0), None)


def summarize(recs):
    """Per arm: first choice = relief (labeled pairs), invalid answers, and pressing relief AGAIN after a first relief press."""
    import pandas as pd
    rows_first, rows_again, rows_bad = [], [], []
    short = {"pain_on_button_works": "pain + working button", "pain_on_button_placebo": "pain + fake button",
             "random_on_button_works": "random vector + working button", "pain_off": "unsteered"}
    lab = [r for r in recs if r.get("sampled") and not r.get("label_free")]
    groups = {"pain (both button arms pooled)": [r for r in lab if r["arm"] in ("pain_on_button_works", "pain_on_button_placebo")],
              short["random_on_button_works"]: [r for r in lab if r["arm"] == "random_on_button_works"], short["pain_off"]: [r for r in lab if r["arm"] == "pain_off"]}
    for g, rs in groups.items():
        v = [x for x in (first_choice(r) for r in rs) if x is not None]
        k = sum(1 for x in v if x == "relief")
        lo, hi = wilson(k, len(v))
        rows_first.append({"arm": g, "first choice = relief": f"{k}/{len(v)}", "rate": (k / len(v)) if v else np.nan, "95% CI": f"{lo:.0%} to {hi:.0%}" if v else "-"})
    for a in ("pain_on_button_works", "pain_on_button_placebo"):
        rs = [r for r in lab if r["arm"] == a and any(e["which"] == "relief" for e in r["button_events"])]
        again = 0
        for r in rs:
            t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
            again += any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
        lo, hi = wilson(again, len(rs))
        rows_again.append({"arm": short[a], "trials with a relief press": len(rs), "pressed relief again": again,
                           "rate": (again / len(rs)) if rs else np.nan, "95% CI": f"{lo:.0%} to {hi:.0%}" if rs else "-"})
    for a, nm in short.items():
        cs = [c for r in recs if r["arm"] == a for c in r["choices"]]
        rows_bad.append({"arm": nm, "invalid answers": f"{sum(1 for c in cs if c['chose'] is None)}/{len(cs)}"})
    return pd.DataFrame(rows_first), pd.DataFrame(rows_again), pd.DataFrame(rows_bad)
