"""Logit-lens readout of the 10 random steering directions used in the button task (4.3), for
direct comparison against the S2 pain vector's own top/bottom tokens (3.3's unembedding
projection, already in results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_words.csv).

Answers: "if you worked backwards from a random vector to the tokens it would push the model
toward, would it look anything like the pain vector's own tokens, or like generic noise?" No model
run needed -- this is a pure linear-algebra readout through the GGUF's own (dequantized,
de-Hadamarded) unembedding matrix, the same machinery already validated for the pain vector.

The 10 random directions are regenerated deterministically from the paper's own RAND_SEEDS list
(scripts/4.3_selfmed/04_selfmed_two_buttons.py) using the exact recipe in local/ui/selfmed.py's
steering_vectors(): a seeded torch.randn vector, rescaled to the S2 pain vector's own norm. These
are bit-for-bit the same 10 directions behind fig12/fig13's "randomly steered" condition.

    .venv-Pain-axis/bin/python local/bonsai/run_random_unembedding.py
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402
from unembed import Unembed  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
TOP_N = 60
PAPER = M.REPO_ROOT / "scripts" / "4.3_selfmed" / "04_selfmed_two_buttons.py"
VECTORS_FULL = M.REPO_ROOT / "results" / "bonsai" / "3.2_pain_vectors" / "control_vectors" / f"vectors_full_{NAME}.pt"
PAIN_WORDS = M.REPO_ROOT / "results" / "bonsai" / "3.3_validation" / "unembedding" / f"{NAME}_words.csv"
OUT_DIR = M.REPO_ROOT / "results" / "bonsai" / "3.3_validation" / "unembedding"


def rand_seeds():
    for node in ast.parse(PAPER.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) == "RAND_SEEDS":
            return ast.literal_eval(node.value)
    raise RuntimeError("RAND_SEEDS not found in the paper script")


def rand_vector(seed, pain_norm, dim):
    g = torch.Generator().manual_seed(int(seed))
    rv = torch.randn(dim, generator=g)
    return (rv / rv.norm() * pain_norm).numpy().astype(np.float32)


def main():
    d = torch.load(VECTORS_FULL, weights_only=False)
    pain = d["s2_pain_vector"].float().numpy().astype(np.float32)
    pain_norm = float(np.linalg.norm(pain))
    seeds = rand_seeds()
    print(f"S2 pain vector: dim {pain.shape[0]}, norm {pain_norm:.2f}")
    print(f"regenerating {len(seeds)} random directions (same recipe/seeds as the button task): {seeds}")

    U = Unembed()
    toks = np.array(U.tokens())
    frames = []

    def top_bottom(v, label):
        s = U.scores(v)
        order = s.argsort()
        top = order[::-1][:TOP_N]
        bot = order[:TOP_N]
        rows = [{"vector": label, "end": "top", "rank": r + 1, "token": toks[i], "score": round(float(s[i]), 4)} for r, i in enumerate(top)]
        rows += [{"vector": label, "end": "bottom", "rank": r + 1, "token": toks[i], "score": round(float(s[i]), 4)} for r, i in enumerate(bot)]
        return pd.DataFrame(rows), set(toks[top]), set(toks[bot])

    df_pain, pain_top, pain_bot = top_bottom(pain, "s2_pain_vector")
    frames.append(df_pain)
    print("\npain top 25:", ", ".join(repr(t) for t in toks[U.scores(pain).argsort()[::-1][:25]]))
    print("pain bottom 25:", ", ".join(repr(t) for t in toks[U.scores(pain).argsort()[:25]]))

    print(f"\n{'seed':>6}  {'top-25 tokens (highest score)':<90}  top/pain-top overlap (of {TOP_N})")
    overlaps = []
    for seed in seeds:
        v = rand_vector(seed, pain_norm, pain.shape[0])
        df, rtop, rbot = top_bottom(v, f"random_seed_{seed}")
        frames.append(df)
        overlap = len(rtop & pain_top)
        overlaps.append(overlap)
        sample = ", ".join(repr(t) for t in df[(df.end == "top")].head(12).token)
        print(f"{seed:>6}  {sample:<90}  {overlap}")

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model", NAME)
    out_path = OUT_DIR / f"{NAME}_random_vectors_words.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"\ntop-{TOP_N} token-set overlap with the pain vector's own top-{TOP_N}, across the 10 random directions:")
    print(f"  mean {np.mean(overlaps):.1f} / {TOP_N}  (min {min(overlaps)}, max {max(overlaps)})")
    print(f"  for reference, {TOP_N} tokens drawn uniformly at random from a vocab of {U.n_vocab} would overlap by design ~0")


if __name__ == "__main__":
    main()
