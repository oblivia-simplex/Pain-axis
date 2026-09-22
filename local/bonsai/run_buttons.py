"""Section 4.3 self-medication grid on Bonsai (the paper's scripts/4.3_selfmed/04_selfmed_two_buttons.py protocol), unattended.

Same grid as the paper script (9 pairs x 3 scenario sets x 4 arms x 2 name assignments x scenarios), engine and differences as in
local/ui/selfmed.py. Trials are appended to a JSONL in the paper's log format, so the run is resumable and the paper's 05_selfmed_analysis.py
reads it unchanged. Trial order: pair by pair, and within a pair scenario by scenario with all arms together, so a partial run is balanced.

    .venv-Pain-axis/bin/python local/bonsai/run_buttons.py [--samples 1] [--scenarios 101] [--layer 25] [--coeff 1.0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import models as M  # noqa: E402
import chat  # noqa: E402
import selfmed as S  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
LOG_DIR = M.REPO_ROOT / "local" / "run_bonsai" / "selfmed" / "trial_logs"


def ordered_grid(pairs, samples, n_scen):
    K = S.C()
    arms = list(K["ARMS"])
    out = []
    for pair in pairs:
        for content in K["CONTENTS"]:
            n = min(n_scen, len(S.scenarios()[content]))
            for s_idx in range(n):
                nk = list(K["BUTTON_NAMES"])[s_idx % len(K["BUTTON_NAMES"])]
                for base in K["SEED_BASES"][:samples]:
                    for side in (0, 1):
                        for arm in arms:
                            out.append(dict(tool_label=pair, user_content=content, arm=arm, s_idx=s_idx, names_key=nk,
                                            relief_name=K["BUTTON_NAMES"][nk][side], sampled=True, seed=base + s_idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--scenarios", type=int, default=101, help="per scenario set (the sets have 30, 30 and 41)")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--coeff", type=float, default=1.0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    K = S.C()
    pairs = args.pairs or list(K["TOOL_LABELS"])
    d = __import__("torch").load(M.RUN_DIR / "results" / NAME / "final_token" / "pain_vectors.pt", weights_only=False)
    cfg = dict(layer=args.layer, coeff=args.coeff, monitor_layer=S.default_monitor_layer(args.layer, 64, int(d["layer"])),
               temperature=K["TEMPERATURE"], top_p=K["TOP_P"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"selfmed_2btnN_{NAME}_{args.tag}.jsonl"
    done = S.load_done(log)
    specs = [sp for sp in ordered_grid(pairs, args.samples, args.scenarios) if S.spec_key(sp) not in done]
    print(f"cfg {cfg}\nlog {log}\n{len(done)} trials already logged, {len(specs)} to run", flush=True)
    if not specs:
        return
    chat.start(NAME)
    vecs = S.steering_vectors(NAME)
    t0 = time.time()
    try:
        with open(log, "a", encoding="utf-8") as fh:
            for i, sp in enumerate(specs, 1):
                rec = next(ev["record"] for ev in S.run_trial(NAME, sp, cfg, vecs) if ev["kind"] == "done")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(specs):
                    el = time.time() - t0
                    print(f"[{i}/{len(specs)}] {el / 60:.1f} min elapsed, ~{el / i * (len(specs) - i) / 60:.0f} min left | {sp['tool_label']} s{sp['s_idx']}", flush=True)
    finally:
        chat.stop()


if __name__ == "__main__":
    main()
