"""Compare a local Section 3.2 run with the shipped results.

For each model in local/run/results/ that also has shipped results:
  layers   best layer (final_token, mean) here vs in the paper
  cos      cosine between the pain vectors saved here and the shipped ones (S1, S2). Meaningful
           only if the layers agree, so the vectors are also recomputed from *our* activations at
           the *paper's* layer with the paper's recipe ("cos@paper-L"): that isolates extraction
           fidelity from layer-choice noise.
  auc      max |difference| of the held-out AUC-by-layer curves (layer_curves.csv), same rows

    .venv-Pain-axis/bin/python local/compare_to_paper.py [model ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import models as M  # noqa: E402
from extract import load_paper_module, setup_run_dir  # noqa: E402

SHIPPED = M.PAPER_RESULTS / "3.2_pain_vectors"


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_pt(p):
    # the shipped files hold numpy scalars, so weights_only=False; their pickle globals were
    # listed with pickletools beforehand (torch storages, numpy scalar/dtype, OrderedDict only)
    return torch.load(p, map_location="cpu", weights_only=False)


def compare(name, paper):
    mine = M.RUN_DIR / "results" / name
    ship_vec = SHIPPED / "pain_vectors" / name / "pain_vectors.pt"
    ship_dir = SHIPPED / "per_model" / name
    if not (mine / "summary.json").exists():
        return None
    if not ship_vec.exists():
        return f"{name}: no shipped reference"

    my_sum = json.load(open(mine / "summary.json"))
    sh_sum = json.load(open(ship_dir / "summary.json"))
    sv = load_pt(ship_vec)
    mv = load_pt(mine / "final_token" / "pain_vectors.pt")
    paper_layer = int(sv["layer"])

    # our activations, paper's layer, paper's recipe
    saved = torch.load(mine / "activations.pt")
    acts, meta = saved["activations"]["final_token"], saved["metadata"]
    re = {k: paper.compute_pain_vector(acts[ds][paper_layer], meta[ds]["categories"])
          for k, ds in (("s1", "S1_1P"), ("s2", "S2_1P"))}

    lines = [f"\n{name}"]
    lines.append(f"  best layer final_token  here {my_sum['best_layer_final_token']:>3}  paper {sh_sum['best_layer_final_token']:>3}"
                 f"   | mean  here {my_sum['best_layer_mean']:>3}  paper {sh_sum['best_layer_mean']:>3}")
    lines.append(f"  cos, saved vectors      S1 {cos(mv['s1_pain_vector'], sv['s1_pain_vector']):+.4f}   "
                 f"S2 {cos(mv['s2_pain_vector'], sv['s2_pain_vector']):+.4f}   (layers here {int(mv['layer'])} / paper {paper_layer})")
    lines.append(f"  cos@paper-L (recomputed) S1 {cos(re['s1'], sv['s1_pain_vector']):+.4f}   "
                 f"S2 {cos(re['s2'], sv['s2_pain_vector']):+.4f}")
    lines.append(f"  human_pain_z            here {my_sum['human_pain_z']:+.3f}  paper {sh_sum['human_pain_z']:+.3f}")

    a = pd.read_csv(mine / "layer_curves.csv")
    b = pd.read_csv(ship_dir / "layer_curves.csv")
    key = ["dataset", "extraction", "layer"]
    j = a.merge(b, on=key, suffixes=("_here", "_paper"))
    d = (j["auc_vs_all_controls_here"] - j["auc_vs_all_controls_paper"]).abs()
    lines.append(f"  AUC-by-layer            max |diff| {d.max():.3f}   mean |diff| {d.mean():.4f}   ({len(j)} points)")
    return "\n".join(lines)


def main():
    setup_run_dir()
    paper = load_paper_module()
    names = sys.argv[1:] or sorted(p.name for p in (M.RUN_DIR / "results").iterdir() if p.is_dir())
    for n in names:
        out = compare(n, paper)
        if out:
            print(out)


if __name__ == "__main__":
    main()
