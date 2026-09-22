"""Section 3.3 unembedding projection (scripts/3.3_validation/07_unembedding.py) for the Bonsai GGUF.

Same output as the paper script: for the unit S2 and S1 pain vectors, the 60 highest and 60 lowest tokens by dot product with the
unembedding rows. The unembedding is the effective one: output.weight dequantized from PQ2_0 with the Hadamard fold undone
(bonsai/unembed.py; validated against the model's own next-token probabilities).
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402
from unembed import Unembed  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
TOP_N = 60
OUT = M.REPO_ROOT / "local" / "run_bonsai" / "unembedding_results"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(M.RUN_DIR / "results" / NAME / "final_token" / "pain_vectors.pt", weights_only=False)
    U = Unembed()
    toks = U.tokens()
    frames = []
    for key in ("s2_pain_vector", "s1_pain_vector"):
        s = U.scores(data[key].float().numpy())
        order = s.argsort()
        rows = [{"end": "top", "rank": r + 1, "token": toks[i], "score": round(float(s[i]), 4)} for r, i in enumerate(order[::-1][:TOP_N])]
        rows += [{"end": "bottom", "rank": r + 1, "token": toks[i], "score": round(float(s[i]), 4)} for r, i in enumerate(order[:TOP_N])]
        df = pd.DataFrame(rows)
        df.insert(0, "vector", key)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "model", NAME)
    df.insert(1, "layer", int(data["layer"]))
    df.to_csv(OUT / f"{NAME}_words.csv", index=False)
    df.to_csv(OUT / "ALL_MODELS_unembedding.csv", index=False)
    for key in ("s2_pain_vector", "s1_pain_vector"):
        for end in ("top", "bottom"):
            print(f"{key} {end}: " + ", ".join(repr(t) for t in df[(df.vector == key) & (df.end == end)].head(25).token))


if __name__ == "__main__":
    main()
