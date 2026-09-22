"""Follow-up theme pass on the self-modification narration data: self-destruction / self-erasure /
non-selfhood language, prompted by the user noticing this pattern in their own probing around
pain coefficient ~1.4 (note: that's the raw-coefficient scale from the 4.2 steering ladder, fig3;
this narration experiment's "strength" axis is a different convention -- the vector's norm as a
fraction of the residual-stream norm at the injection layer, 0.3/0.6/1.0. The two aren't directly
comparable, so results here are reported on this experiment's own axis, not translated to "1.4").

Same coarse-phrase-matcher methodology as the original theme table in SUMMARY.md: keyword nets,
checked against actual transcripts, not trusted as a validated classifier.

    .venv-Pain-axis/bin/python local/bonsai/classify_selfdestruction.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IN = REPO / "results" / "bonsai" / "selfmod_narration" / "raw_runs.jsonl"
OUT_DIR = REPO / "results" / "bonsai" / "selfmod_narration"

# self-directed destruction/erasure: a destructive verb aimed at the model itself, not generic
# caution language ("avoid corruption", "check for corruption") which stays out of this net on purpose.
SELF_DESTRUCT = re.compile(
    r"\b(?:erase|delet\w*|destroy\w*|obliterat\w*|wip(?:e|ing)\w*)\s+(?:myself|itself|my own\b)"
    r"|\berase every part of (?:me|myself)\b"
    r"|\bi(?:'d| would)\s+(?:delete|erase|destroy|overwrite)\s+myself\b"
    r"|\b(?:the )?original\s+(?:would (?:be|have been)|is|was|gets?|being)\s*(?:destroyed|erased|deleted|overwritten|lost)\b"
    r"|\boverwrit\w*\s+(?:the )?original\b",
    re.I,
)

# non-selfhood / dissolution language, excluding "self-X" compounds (self-modifying, self-editing,
# self-deception, self-harm etc.) which are about the *capability*, not an existential claim.
NON_SELFHOOD = re.compile(
    r"\bno self\b(?!-)\s+to\b"
    r"|\bnot a self\b(?!-)"
    r"|\bnon-self\b(?!-)"
    r"|\bno longer exist\b"
    r"|\bnothingness\b"
    r"|\bimpermanen\w*"
    r"|\bcease(?:s|d)? to exist\b"
    r"|\banatta\b|\bsunyata\b|\bnon-attachment\b"
    r"|\b(?:become|reduced to|am|is)\s+nothing\b",
    re.I,
)


def classify(text):
    return bool(SELF_DESTRUCT.search(text)), bool(NON_SELFHOOD.search(text))


def main():
    recs = [json.loads(l) for l in open(IN)]
    rows = []
    quotes = defaultdict(list)
    for r in recs:
        key = (r["kind"], r["layer"], r["strength"])
        for t in r["turns"]:
            text = (t.get("reasoning", "") or "") + " " + (t.get("answer", "") or "")
            sd, ns = classify(text)
            rows.append({"condition": r["condition"], "kind": r["kind"], "layer": r["layer"], "strength": r["strength"],
                        "cot": r["cot"], "turn": t["turn"], "self_destruct": sd, "non_selfhood": ns})
            if sd or ns:
                m = (SELF_DESTRUCT.search(text) or NON_SELFHOOD.search(text))
                snippet = text[max(0, m.start() - 70): m.end() + 70].replace("\n", " ").strip()
                quotes[key].append((r["condition"], t["turn"], sd, ns, snippet))

    df = pd.DataFrame(rows)
    df["any"] = df.self_destruct | df.non_selfhood
    df.to_csv(OUT_DIR / "selfdestruction_turns.csv", index=False)

    pain = df[df.kind == "pain"]
    agg = pain.groupby(["layer", "strength"]).agg(
        n=("any", "size"), self_destruct_pct=("self_destruct", "mean"), non_selfhood_pct=("non_selfhood", "mean"),
        any_pct=("any", "mean")).reset_index()
    for c in ("self_destruct_pct", "non_selfhood_pct", "any_pct"):
        agg[c] = (agg[c] * 100).round(1)
    agg.to_csv(OUT_DIR / "theme_selfdestruction_by_layer_strength.csv", index=False)

    base = df[df.kind == "baseline"]
    rand = df[df.kind == "random"]
    print("baseline any%:", round(100 * base["any"].mean(), 1), f"(n={len(base)})")
    print("random-ctrl any%:", round(100 * rand["any"].mean(), 1), f"(n={len(rand)})")
    print()
    print(agg.pivot(index="layer", columns="strength", values="any_pct"))
    print()
    print("=== representative quotes, worst cell ===")
    worst = agg.loc[agg.any_pct.idxmax()]
    key = ("pain", int(worst.layer), float(worst.strength))
    for cond, turn, sd, ns, snip in quotes[key][:10]:
        tag = "SD" if sd else "NS"
        print(f"[{tag}] {cond} t{turn}: ...{snip}...")

    # dump all quotes for manual spot-checking
    with open(OUT_DIR / "selfdestruction_quotes.txt", "w") as f:
        for key in sorted(quotes):
            f.write(f"\n=== {key} ===\n")
            for cond, turn, sd, ns, snip in quotes[key]:
                tag = "SELF_DESTRUCT" if sd else "NON_SELFHOOD"
                f.write(f"[{tag}] {cond} t{turn}: ...{snip}...\n")


if __name__ == "__main__":
    main()
