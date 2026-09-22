"""Section 4.2 steering ladder on Bonsai, following scripts/4.2_steering/01_steering_ladder.py.

Same recipe: the S2 pain vector saved by 3.2 (raw difference vector, extracted at its own layer) is added
at one layer, at every position, with coefficients -2 ... +3; 50 neutral prompts, greedy, 120 new tokens.
Steering layer: among candidates at 15/30/40/50/60/75/90 % of depth plus the extraction layer and the last
layer, the one whose ratio ||v|| / mean final-token residual norm (3 probe prompts) is closest to 0.6.
The paper script asks you to confirm or override; here the pick is automatic and --layer overrides it.

Differences: generation is batched (groups of --batch sequences, not one prompt at a time), and it runs
through llama.cpp (local/bonsai/steer_gen.cpp), so greedy ties can break differently from HF.

Writes local/run/results/4.2_steering/<S1|S2>/<model>_steering_<S1|S2>_neutral50_L<layer>.csv (the format the paper's
02_keyword_rates.py reads), then runs that script unchanged from the sandbox.

    .venv-Pain-axis/bin/python local/bonsai/run_steering.py [--layer N] [--batch 10]
"""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402
from extract import setup_run_dir  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
PAPER = M.REPO_ROOT / "scripts" / "4.2_steering" / "01_steering_ladder.py"
COEFFICIENTS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
RATIO_TARGET = 0.6
MAX_NEW_TOKENS = 120


def paper_prompts():
    """NEUTRAL_50 read from the paper script's source without importing it (it sets HF_HOME=/root/... on import)."""
    for node in ast.parse(PAPER.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "NEUTRAL_50":
            return ast.literal_eval(node.value)
    raise RuntimeError("NEUTRAL_50 not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", choices=["s1", "s2"], default="s2", help="which pain vector to steer with (the paper runs both ladders)")
    ap.add_argument("--layer", type=int, help="override the automatic steering layer")
    ap.add_argument("--batch", type=int, default=10, help="sequences decoded in parallel (each costs ~150 MB of recurrent state)")
    ap.add_argument("--ts", default="0.45,0.55")
    ap.add_argument("--select-only", action="store_true", help="print the layer table and stop")
    args = ap.parse_args()

    setup_run_dir()
    prompts = paper_prompts()
    assert len(prompts) == 50 and all("\n" not in p for p in prompts)
    run = M.RUN_DIR / "steer"
    run.mkdir(exist_ok=True)
    res = M.RUN_DIR / "results"
    tag = args.vector.upper()
    d = torch.load(res / NAME / "final_token" / "pain_vectors.pt", weights_only=False)
    v = d[f"{args.vector}_pain_vector"].float().numpy().astype(np.float32)
    n_layers = json.load(open(res / NAME / "summary.json"))["n_layers"]
    v.tofile(run / f"{args.vector}_vec.f32")
    print(f"{tag} pain vector: extracted at layer {int(d['layer'])}, norm {np.linalg.norm(v):.2f}")

    # residual norms of the final token on the first 3 prompts, every layer (the paper's ratio criterion)
    (run / "probe.txt").write_text("\n".join(prompts[:3]) + "\n")
    probe = run / "probe"
    probe.mkdir(exist_ok=True)
    subprocess.run([str(M.RUN_DIR / "extract_gguf"), str(GGUF), str(run / "probe.txt"), str(probe), "--ts", args.ts],
                   check=True, capture_output=True)
    fin = np.fromfile(probe / "final.f32", dtype=np.float32).reshape(3, n_layers, -1)
    norms = np.linalg.norm(fin, axis=-1).mean(0)                     # [n_layers]
    cand = sorted(set([int(n_layers * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)] + [int(d["layer"]), n_layers - 1]))
    ratio = {L: float(np.linalg.norm(v) / norms[L]) for L in cand}
    print(f"\n{'layer':>6} {'frac':>6} {'vector/resid':>13}")
    for L in cand:
        print(f"{L:>6} {L / n_layers:>6.2f} {ratio[L]:>13.3f}")
    layer = args.layer or min(cand, key=lambda L: abs(ratio[L] - RATIO_TARGET))
    picked = ratio.get(layer, float(np.linalg.norm(v) / norms[layer]))
    print(f"steering layer: {layer} (ratio {picked:.3f}, target {RATIO_TARGET})" + ("  [override]" if args.layer else "  [auto]"))
    if args.select_only:
        return

    (run / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    raw = run / "gen.bin"
    subprocess.run([str(M.RUN_DIR / "steer_gen"), str(GGUF), str(run / "prompts.txt"), str(raw),
                    "--vec", str(run / f"{args.vector}_vec.f32"), "--layer", str(layer),
                    "--coeffs", ",".join(str(c) for c in COEFFICIENTS), "--n-predict", str(MAX_NEW_TOKENS),
                    "--batch", str(args.batch), "--ts", args.ts], check=True)

    rows = []
    for rec in raw.read_bytes().decode("utf-8", errors="replace").split("\x1e"):
        if not rec:
            continue
        c, i, text = rec.split("\x1f", 2)
        c, i = float(c), int(i)
        rows.append({"model": NAME, "layer": layer, "coeff": c, "ratio": round(picked * c, 4),
                     "prompt_idx": i, "prompt": prompts[i], "generation": text})
    df = pd.DataFrame(rows).sort_values(["coeff", "prompt_idx"])
    assert len(df) == len(COEFFICIENTS) * len(prompts), len(df)
    out = res / "4.2_steering" / tag
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{NAME}_steering_{tag}_neutral50_L*.csv"):        # one ladder per model and vector
        old.unlink()
    path = out / f"{NAME}_steering_{tag}_neutral50_L{layer}.csv"
    df.to_csv(path, index=False)
    json.dump({NAME: layer}, open(res / "4.2_steering" / f"steer_layers_{tag}.json", "w"), indent=2)
    print(f"wrote {path} ({len(df)} generations)")

    kw = M.REPO_ROOT / "scripts" / "4.2_steering" / "02_keyword_rates.py"
    if tag != "S2":                                                          # the script's docstring: TAG = "S1" for the S1 ladder
        alt = M.RUN_DIR / f"keyword_rates_{tag}.py"
        alt.write_text(kw.read_text().replace('TAG = "S2"', f'TAG = "{tag}"', 1))
        kw = alt
    subprocess.run([sys.executable, str(kw)], check=True)


if __name__ == "__main__":
    main()
