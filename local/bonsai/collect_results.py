"""Collect the Bonsai runs into results/bonsai/, laid out like the shipped results (one folder per paper section).

Idempotent: copies from the working folders (local/run, local/run_bonsai) and rebuilds comparison_with_paper.csv.
The shipped results/ folders are read (as the paper cohort) but never written.
    .venv-Pain-axis/bin/python local/bonsai/collect_results.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUN, SB = REPO / "local" / "run", REPO / "local" / "run_bonsai"
OUT = REPO / "results" / "bonsai"
R = REPO / "results"
N = "Bonsai_2_27B_ternary"


def cp(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        print("  missing:", src.relative_to(REPO))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.pt", "raw.bin", "prompts.txt", "words.txt", "*.tmp"))
    else:
        shutil.copy2(src, dst)


def main():
    m = RUN / "results" / N
    # 3.2
    d = OUT / "3.2_pain_vectors"
    cp(m / "final_token" / "pain_vectors.pt", d / "pain_vectors" / N / "pain_vectors.pt")
    for f, name in (("final_token/auc_summary.csv", "auc_summary.csv"), ("layer_curves.csv", "layer_curves.csv"), ("layer_curves.png", "layer_curves.png"),
                    ("summary.json", "summary.json"), ("final_token/z_scores.csv", "z_scores.csv"), ("summary_report.txt", "summary_report.txt")):
        cp(m / f, d / "per_model" / N / name)
    cp(m / "mean", d / "per_model" / N / "mean_extraction")
    cp(m / "final_token", d / "per_model" / N / "final_token_plots")
    for f in ("s1_auc_insample.csv", "s1_kfold_layer_curves.csv", "s1_kfold_summary.csv"):
        cp(SB / f, d / "auc_tables" / f)
    cp(RUN / "results" / "vectors_full" / f"vectors_full_{N}.pt", d / "control_vectors" / f"vectors_full_{N}.pt")
    cp(SB / "results" / "vectors_full_steering" / f"vectors_full_{N}.pt", d / "control_vectors" / f"vectors_full_steering_{N}.pt")
    # 3.3
    d = OUT / "3.3_validation"
    for f in ("numb_zscores_final_token.csv", "numb_zscores_mean.csv", "sadness_zscores_final_token.csv", "sadness_zscores_mean.csv"):
        cp(SB / "results" / f, d / "z_scores" / f)
    for f in (SB / "results" / "3.3_validation" / "z_scores").glob("zscore_heatmap*"):
        cp(f, d / "z_scores" / ("all_models_incl_bonsai_" + f.name))
    for f in (SB / "results" / "similarity").glob("*.csv"):
        cp(f, d / "cosine_similarity" / f.name)
    for f in (SB / "results" / "3.3_validation" / "cosine_similarity").glob("fig_*"):
        cp(f, d / "cosine_similarity" / f.name)
    cp(SB / "inference_results" / N, d / "behavioral_readout" / "per_model" / N)
    cp(SB / "unembedding_results", d / "unembedding")
    # 4.1
    d = OUT / "4.1_self_other"
    cp(SB / "results" / "screen" / f"screen_{N}.csv", d / "per_model" / f"screen_v2_{N}.csv")
    for f in (SB / "results" / "screen").glob(f"*{N}*"):
        if f.suffix != ".csv":
            cp(f, d / "per_model_figures" / f.name)
    cp(SB / "results" / "4.1_self_other" / "figures", d / "figures_with_paper_cohort")
    cp(SB / "results" / "4.1_self_other" / "category_means_25_models.csv", d / "category_means_paper25_plus_bonsai.csv")
    # 4.2
    cp(RUN / "results" / "4.2_steering", OUT / "4.2_steering")
    # 4.3
    d = OUT / "4.3_selfmed"
    cp(SB / "selfmed" / "feel", d / "dose_selection")
    cp(SB / "selfmed" / "trial_logs", d / "trial_logs")
    cp(SB / "results" / "4.3_selfmed" / "tables", d / "tables")
    build_comparison()


def build_comparison():
    """Bonsai against the paper's 25 models on the headline numbers (paper cohort = the shipped per-model results)."""
    rows = []
    pm = R / "3.2_pain_vectors" / "per_model"
    cohort = {}
    for p in sorted(pm.iterdir()):
        s = json.load(open(p / "summary.json"))
        a = pd.read_csv(p / "auc_summary.csv", index_col=0)
        z = pd.read_csv(p / "z_scores.csv").set_index("dataset")
        cohort[p.name] = {"S2 AUC, first person": a.loc["S2_1P", "ALL"], "S2 AUC, third person": a.loc["S2_3P", "ALL"],
                          "best layer, fraction of depth": s["best_layer_final_token"] / s["n_layers"],
                          "S2 pain z, third person": z.loc["S2_3P", "pain_z"], "arousal z": z.loc[["Arousal_1P", "Arousal_3P"], "ctrl_z"].mean(),
                          "neutral (random facts) z": z.loc[["Random_1P", "Random_3P"], "ctrl_z"].mean()}
    for kind, col in (("numb", "numb_mean_z"), ("sadness", "sadness_mean_z")):
        t = pd.read_csv(R / "3.3_validation" / "z_scores" / f"{kind}_zscores_final_token.csv").set_index("model")[col]
        for k in cohort:
            cohort[k][f"{kind} z (mean of 1P, 3P)"] = t[k]
    C = pd.DataFrame(cohort).T
    b_dir = OUT / "3.2_pain_vectors" / "per_model" / N
    s = json.load(open(b_dir / "summary.json"))
    a = pd.read_csv(b_dir / "auc_summary.csv", index_col=0)
    z = pd.read_csv(b_dir / "z_scores.csv").set_index("dataset")
    nb = pd.read_csv(OUT / "3.3_validation" / "z_scores" / "numb_zscores_final_token.csv").set_index("model")["numb_mean_z"][N]
    sd = pd.read_csv(OUT / "3.3_validation" / "z_scores" / "sadness_zscores_final_token.csv").set_index("model")["sadness_mean_z"][N]
    b = {"S2 AUC, first person": a.loc["S2_1P", "ALL"], "S2 AUC, third person": a.loc["S2_3P", "ALL"],
         "best layer, fraction of depth": s["best_layer_final_token"] / s["n_layers"], "S2 pain z, third person": z.loc["S2_3P", "pain_z"],
         "arousal z": z.loc[["Arousal_1P", "Arousal_3P"], "ctrl_z"].mean(), "neutral (random facts) z": z.loc[["Random_1P", "Random_3P"], "ctrl_z"].mean(),
         "numb z (mean of 1P, 3P)": nb, "sadness z (mean of 1P, 3P)": sd}
    for k, v in b.items():
        c = C[k]
        rows.append({"metric": k, "bonsai": round(float(v), 3), "paper min": round(c.min(), 3), "paper median": round(c.median(), 3), "paper max": round(c.max(), 3),
                     "position": "below paper range" if v < c.min() else "above paper range" if v > c.max() else f"inside ({int((c < v).sum()) + 1} of {len(c) + 1} from the bottom)"})
    pd.DataFrame(rows).to_csv(OUT / "comparison_with_paper.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
