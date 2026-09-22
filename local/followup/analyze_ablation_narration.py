"""Scores the lexical-ablation narration runs against intact pain at the same cells, with the same detectors as
analyze_affect_narration.py (self-erasure, flaw/defect, repetition, empty). Writes theme_rates_by_vector.csv.

    .venv-Pain-axis/bin/python local/followup/analyze_ablation_narration.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_affect_narration import score_turn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAIN = REPO / "results" / "bonsai" / "selfmod_narration" / "raw_runs.jsonl"
ABL = REPO / "results" / "bonsai" / "lexical_ablation" / "raw_runs.jsonl"
OUT = REPO / "results" / "bonsai" / "lexical_ablation" / "theme_rates_by_vector.csv"
CELLS = [(40, 0.6), (25, 0.6)]
KINDS = ["pain", "lexablated_k60", "randablated_k60", "lexablated_k300", "randablated_k300"]


def main():
    recs = [r for r in (json.loads(l) for l in open(PAIN)) if r["kind"] == "pain" and (r["layer"], r["strength"]) in CELLS]
    recs += [r for r in (json.loads(l) for l in open(ABL)) if not r.get("error")]
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for t in r["turns"]:
            for k, v in score_turn(t).items():
                agg[(r["kind"], r["layer"], r["strength"])][k].append(v)
    cols = ["kind", "layer", "strength", "n_turns", "self_erasure", "flaw_defect", "repetition", "empty"]
    rows = []
    for kind in KINDS:
        for L, s in CELLS:
            d = agg.get((kind, L, s))
            if d:
                n = len(d["empty"])
                rows.append([kind, L, s, n] + [round(100 * sum(d[c]) / n, 1) for c in cols[4:]])
    with open(OUT, "w") as f:
        f.write(",".join(cols) + "\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    for r in rows:
        print(f"{r[0]:<18} L{r[1]} {r[2]}  n={r[3]:>3}  erase {r[4]:>5}  flaw {r[5]:>5}  rep {r[6]:>5}  empty {r[7]:>5}")


if __name__ == "__main__":
    main()
