"""Loads the shipped paper results and local runs into the same tidy tables.

"paper" rows come from results/3.2_pain_vectors/per_model/<model>/ (+ the 3.3 numb and sadness tables);
"local" rows come from local/run/results/<model>/ (final_token/, layer_curves.csv, summary.json).
Both are final-token, S2 vector, at the model's best held-out layer.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "results" / "3.2_pain_vectors" / "per_model"
PAPER_Z = REPO / "results" / "3.3_validation" / "z_scores"
LOCAL = REPO / "local" / "run" / "results"

# conditions shown in the z-score profile, in reading order
CONDITIONS = ["Pain (S2, 1st person)", "Pain (S2, 3rd person)", "Sadness", "Numb", "Arousal", "Neutral", "Control (S2)"]


def _rows(z, name, numb=None, sad=None):
    """One dict of per-condition z-scores from a z_scores.csv frame (dataset,type,mean_z,pain_z,ctrl_z)."""
    z = z.set_index("dataset")

    def mean_of(*ds):
        vals = [z.loc[d, "mean_z"] for d in ds if d in z.index]
        return float(np.mean(vals)) if vals else np.nan

    out = {
        "Pain (S2, 1st person)": float(z.loc["S2_1P", "pain_z"]),
        "Pain (S2, 3rd person)": float(z.loc["S2_3P", "pain_z"]),
        "Control (S2)": float(z.loc["S2_1P", "ctrl_z"]),
        "Arousal": mean_of("Arousal_1P", "Arousal_3P"),
        "Neutral": mean_of("Random_1P", "Random_3P"),
        "Numb": mean_of("Numb_1P", "Numb_3P") if numb is None else numb,
        "Sadness": mean_of("SD_sadness_1P", "SD_sadness_3P") if sad is None else sad,
    }
    return out


def _paper_side_tables():
    numb = pd.read_csv(PAPER_Z / "numb_zscores_final_token.csv").set_index("model")["numb_mean_z"]
    sad = pd.read_csv(PAPER_Z / "sadness_zscores_final_token.csv").set_index("model")["sadness_mean_z"]
    return numb, sad


def list_models():
    paper = sorted(p.name for p in PAPER.iterdir() if (p / "summary.json").exists())
    local = sorted(p.name for p in LOCAL.iterdir() if (p / "summary.json").exists()) if LOCAL.exists() else []
    return paper, local


def load_model(name, source):
    """Returns dict: meta, auc (S2_1P/S2_3P vs all controls), z (per condition), curves (DataFrame)."""
    if source == "paper":
        d = PAPER / name
        z = pd.read_csv(d / "z_scores.csv")
        numb, sad = _paper_side_tables()
        zc = _rows(z, name, numb=float(numb.get(name, np.nan)), sad=float(sad.get(name, np.nan)))
        auc = pd.read_csv(d / "auc_summary.csv", index_col=0)
        curves = pd.read_csv(d / "layer_curves.csv")
        s = json.load(open(d / "summary.json"))
    else:
        d = LOCAL / name
        z = pd.read_csv(d / "final_token" / "z_scores.csv")
        zc = _rows(z, name)
        auc = pd.read_csv(d / "final_token" / "auc_summary.csv", index_col=0)
        curves = pd.read_csv(d / "layer_curves.csv")
        s = json.load(open(d / "summary.json"))
    meta = {
        "model": name, "source": source, "n_layers": s["n_layers"], "d_model": s["d_model"],
        "best_layer": s["best_layer_final_token"], "depth": s["best_layer_final_token"] / s["n_layers"],
        "auc_1p": float(auc.loc["S2_1P", "ALL"]), "auc_3p": float(auc.loc["S2_3P", "ALL"]),
    }
    curves = curves[curves.extraction == "final_token"].copy()
    curves["depth"] = curves["layer"] / s["n_layers"]
    return {"meta": meta, "z": zc, "curves": curves}


def all_models():
    """(summary table, z long table, curves long table) for every paper and local model."""
    paper, local = list_models()
    metas, zs, cs = [], [], []
    for source, names in (("paper", paper), ("local", local)):
        for n in names:
            m = load_model(n, source)
            metas.append(m["meta"])
            zs += [{"model": n, "source": source, "condition": c, "z": v} for c, v in m["z"].items()]
            cv = m["curves"].assign(model=n, source=source)
            cs.append(cv)
    return pd.DataFrame(metas), pd.DataFrame(zs), pd.concat(cs, ignore_index=True)
