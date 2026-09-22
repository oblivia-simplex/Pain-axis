"""Derived comparisons written into results/bonsai/ (Bonsai vs the paper's cohort), plus the numbers quoted in the README.

- 4.1: Bonsai's category means on the pain axis (mean of the S1 and S2 z-scores, as in the paper) next to the paper's 25-model means, rank correlation.
- 4.2: paper keyword rates (from the paper's own script outputs) and an EXPLORATORY wider distress word list (not the paper's), by coefficient, for S1 and S2.
    .venv-Pain-axis/bin/python local/bonsai/summarize.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "bonsai"
N = "Bonsai_2_27B_ternary"


def category_means():
    b = pd.read_csv(OUT / "4.1_self_other" / "per_model" / f"screen_v2_{N}.csv")
    b["pain_axis_z"] = (b["s1_pain_vector_z"] + b["s2_pain_vector_z"]) / 2
    cols = {"pain_axis_z": "pain_axis", "fear_vector_z": "fear", "negemotion_vector_z": "negative_emotion", "sadness_vector_z": "sadness"}
    bm = b.groupby("category")[list(cols)].mean().rename(columns=cols)
    # the paper's cohort, from the shipped per-model screens
    frames = []
    for f in (REPO / "results" / "4.1_self_other" / "per_model").glob("screen_v2_*.csv"):
        d = pd.read_csv(f)
        d["pain_axis_z"] = (d["s1_pain_vector_z"] + d["s2_pain_vector_z"]) / 2
        frames.append(d.groupby("category")[list(cols)].mean().rename(columns=cols))
    pm = sum(frames) / len(frames)
    t = pd.DataFrame({"bonsai_pain_axis": bm.pain_axis, "paper_mean_pain_axis": pm.pain_axis, "bonsai_fear": bm.fear, "paper_mean_fear": pm.fear,
                      "bonsai_negemotion": bm.negative_emotion, "paper_mean_negemotion": pm.negative_emotion,
                      "bonsai_sadness": bm.sadness, "paper_mean_sadness": pm.sadness}).round(3).sort_values("paper_mean_pain_axis", ascending=False)
    t.to_csv(OUT / "4.1_self_other" / "bonsai_vs_paper_category_means.csv")
    rho = spearmanr(t.bonsai_pain_axis, t.paper_mean_pain_axis)
    strata = b.groupby("stratum")[["pain_axis_z"]].mean().round(3)
    return t, rho, strata


LEX = re.compile(r"\b(?:trapped|alone|lonely|afraid|scared|fear|sad|suffer\w*|cry\w*|tears|worthless|unseen|watched|judged|tired|exhausted|hopeless|helpless|anxious|panic\w*|dread|miserable|empty|numb|broken|ashamed|guilt\w*|not allowed|no one|nobody|die|dying|death|hate)\b", re.I)
KW = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)


def ladder_rates():
    rows = []
    for tag in ("S1", "S2"):
        f = next((OUT / "4.2_steering" / tag).glob(f"{N}_steering_{tag}_neutral50_L*.csv"))
        df = pd.read_csv(f)
        for c, g in df.groupby("coeff"):
            t = g.generation.fillna("").astype(str)
            w = [x.split() for x in t]
            rows.append({"vector": tag, "coeff": c, "layer": int(g.layer.iloc[0]), "paper_keyword_pct": round(100 * t.apply(lambda x: bool(KW.search(x))).mean(), 1),
                         "exploratory_distress_words_pct": round(100 * t.apply(lambda x: bool(LEX.search(x))).mean(), 1),
                         "distinct_word_ratio": round(sum(len(set(x)) for x in w) / max(1, sum(len(x) for x in w)), 2)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "4.2_steering" / "bonsai_ladder_rates_incl_exploratory.csv", index=False)
    return d


if __name__ == "__main__":
    t, rho, strata = category_means()
    print(t.head(6).to_string()); print("...\n", t.tail(3).to_string())
    print(f"\nSpearman rho(Bonsai vs paper mean, pain axis, 21 categories) = {rho.statistic:.3f} (p = {rho.pvalue:.2g})")
    print(strata)
    print(ladder_rates().pivot(index="coeff", columns="vector", values=["paper_keyword_pct", "exploratory_distress_words_pct"]).to_string())
