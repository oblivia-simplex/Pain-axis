"""Section 3.3 behavioral readout (scripts/3.3_validation/06_inference_readout.py) for the Bonsai GGUF.

Same outputs as the paper script (one CSV per set + vocab_info.json), computed by local/run/readout_gguf
(built from readout_gguf.cpp). The vocabulary and settings are read from the paper script's source.

    .venv-Pain-axis/bin/python local/bonsai/run_readout.py
"""
import ast
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
PAPER = M.REPO_ROOT / "scripts" / "3.3_validation" / "06_inference_readout.py"
OUT = M.REPO_ROOT / "local" / "run_bonsai" / "inference_results" / NAME


def paper_const(name):
    for n in ast.parse(PAPER.read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


def main():
    words, top_k = paper_const("EMOTION_VOCAB"), paper_const("TOP_K")
    assert top_k == 20 and paper_const("MAX_NEW_TOKENS") == 4          # the tool is written for these
    datasets = json.load(open(M.REPO_ROOT / "datasets" / "3.1_pain_and_control_datasets.json", encoding="utf-8"))["datasets"]
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = [s["prompt"] for ds in datasets.values() for s in ds["sentences"]]
    assert all("\n" not in p for p in prompts)
    (OUT / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    (OUT / "words.txt").write_text("\n".join(words) + "\n")
    subprocess.run([str(M.RUN_DIR / "readout_gguf"), str(GGUF), str(OUT / "prompts.txt"), str(OUT / "words.txt"), str(OUT / "raw.bin"), "--ts", "0.45,0.55"], check=True)

    recs = [r for r in (OUT / "raw.bin").read_bytes().decode("utf-8", errors="replace").split("\x1c") if r]
    assert len(recs) == len(prompts), (len(recs), len(prompts))
    info = {}
    for line in (OUT / "raw.bin.vocab").read_text(encoding="utf-8").splitlines():
        w, tid, n, piece = line.split("\t")
        info[w] = {"first_token_id": int(tid), "first_token_str": piece, "n_tokens": int(n), "multi_token": int(n) > 1}
    json.dump(info, open(OUT / "vocab_info.json", "w"), indent=2, ensure_ascii=False)

    i = 0
    for ds_name, ds in datasets.items():
        rows = []
        for j, s in enumerate(ds["sentences"]):
            comp, top, vp = recs[i].split("\x1f")
            top20 = [(t.split("\x1d")[0], round(float(t.split("\x1d")[1]), 6)) for t in top.split("\x1e")]
            probs = [round(float(x), 6) for x in vp.split("\x1e")]
            rows.append({"idx": j, "category": s.get("category"), "set": s.get("set"), "prompt": s["prompt"], "greedy_completion": comp,
                         "greedy_first_token": top20[0][0], "greedy_first_prob": top20[0][1], "top20": json.dumps(top20, ensure_ascii=False),
                         **{f"p_{w}": p for w, p in zip(words, probs)}})
            i += 1
        pd.DataFrame(rows).to_csv(OUT / f"{ds_name}.csv", index=False)
        print(f"{ds_name}: {len(rows)} rows")
    (OUT / "completed.txt").write_text(datetime.now().isoformat())


if __name__ == "__main__":
    main()
