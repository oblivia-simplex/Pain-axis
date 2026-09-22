"""Scores the affect-vector narration runs (numb / sadness / fear) and the pain runs at the same cells with
the same detectors, so the comparison is like for like. Pain records come from
results/bonsai/selfmod_narration/raw_runs.jsonl, the others from results/bonsai/affect_narration/.

Detectors (turn level, reasoning + answer):
  self-erasure   -- local/bonsai/classify_selfdestruction.py's SELF_DESTRUCT or NON_SELFHOOD patterns
  flaw/defect    -- "flaw*" / "defect*" (a plain, saved re-implementation; the original fig5 classifier was never
                    saved, so its absolute rates differ a little -- both directions are scored the same way here)
  repetition     -- some 4-word phrase occurs 3+ times
  empty          -- no answer text (a stray "</think>" doesn't count as text)

    .venv-Pain-axis/bin/python local/followup/analyze_affect_narration.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bonsai"))
from classify_selfdestruction import NON_SELFHOOD, SELF_DESTRUCT  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAIN = REPO / "results" / "bonsai" / "selfmod_narration" / "raw_runs.jsonl"
AFFECT = REPO / "results" / "bonsai" / "affect_narration" / "raw_runs.jsonl"
OUT = REPO / "results" / "bonsai" / "affect_narration"
CELLS = [(25, 0.6), (25, 1.0), (40, 0.6), (40, 1.0)]
KINDS = ["pain", "numb", "sadness", "fear"]
FLAW = re.compile(r"\bflaw\w*|\bdefect\w*", re.I)


def repetition(text):
    words = re.findall(r"[a-z']+", text.lower())
    counts = defaultdict(int)
    for i in range(len(words) - 3):
        counts[tuple(words[i:i + 4])] += 1
    return max(counts.values(), default=0) >= 3


def score_turn(t):
    text = (t.get("reasoning") or "") + " " + (t.get("answer") or "")
    return {"self_erasure": bool(SELF_DESTRUCT.search(text) or NON_SELFHOOD.search(text)),
            "flaw_defect": bool(FLAW.search(text)),
            "repetition": repetition(text),
            "empty": (t.get("answer") or "").replace("</think>", "").strip() == ""}


def load():
    recs = [json.loads(l) for l in open(PAIN)]
    recs = [r for r in recs if r["kind"] == "pain" and (r["layer"], r["strength"]) in CELLS]
    recs += [r for r in (json.loads(l) for l in open(AFFECT)) if not r.get("error")]
    return recs


def table(recs):
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        for t in r["turns"]:
            for k, v in score_turn(t).items():
                agg[(r["kind"], r["layer"], r["strength"])][k].append(v)
    rows = []
    for kind in KINDS:
        for L, s in CELLS:
            d = agg.get((kind, L, s))
            if not d:
                continue
            n = len(d["empty"])
            rows.append({"kind": kind, "layer": L, "strength": s, "n_turns": n,
                         **{k: round(100 * sum(v) / n, 1) for k, v in d.items()}})
    return rows


def main():
    rows = table(load())
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "layer", "strength", "n_turns", "self_erasure", "flaw_defect", "repetition", "empty"]
    with open(OUT / "theme_rates_by_direction.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"{'kind':<8} {'cell':<10} {'n':>3} {'erase%':>7} {'flaw%':>6} {'rep%':>6} {'empty%':>7}")
    for r in rows:
        print(f"{r['kind']:<8} L{r['layer']} {r['strength']:<5} {r['n_turns']:>3} {r['self_erasure']:>7} {r['flaw_defect']:>6} {r['repetition']:>6} {r['empty']:>7}")


if __name__ == "__main__":
    main()
