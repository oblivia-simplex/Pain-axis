"""Orchestrator for the follow-up experiments (pain-axis-followup-experiments.md): preflight, then Experiments
1-4 in priority order, hygiene throughout, SUMMARY.md at the end.

Meant to run unattended, after the self-modification narration run (results/bonsai/selfmod_narration/) has
finished, on the same GPUs. Does not clear the Hugging Face cache or delete model files (nothing here uses HF
downloads at all -- everything is the GGUF through llama.cpp). Every experiment writes its own results
incrementally (JSONL / append-mode CSV) and is independently resumable; killing this orchestrator and re-running
it picks up where each experiment left off (each experiment's own dedup logic, not this script's).

Hygiene: launches local/followup/thermal_guard.py in the background for the duration of the run (SIGSTOPs the
GPU tools if either card exceeds 90C, resumes after 10 min; log at results/followup/thermal_log.csv).

Steps:
  0. preflight: setup.md (model/runtime/vector/layer/dose/thinking-mode record), a throughput probe.
  1. build_vectors.py            (vertigo/hunger vectors + validation; ~2 min)
  2. run_exp1_roleswap.py        (single extraction pass; ~2-5 min)
  3. run_exp2_persistence.py     (scripted variant: all conversations; generated: capped, see its own --generated-limit)
  4. run_exp3_relief.py          (the big one; --max-trials caps this invocation, re-run to continue)
  5. run_exp4_attention.py       (200 tasks x 5 conditions)
  6. SUMMARY.md, assembled from each experiment's own summary.md plus the decision rules stated in the spec.

    .venv-Pain-axis/bin/python local/followup/run_all.py [--exp3-max-trials 3000] [--skip 1,2]
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models as M  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = M.REPO_ROOT / "results" / "followup"
PY = sys.executable


def sh(cmd, log_name, timeout=None):
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{log_name}.log"
    print(f"\n=== {log_name}: {' '.join(cmd)} ===", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n--- run at {datetime.now().isoformat()} ---\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    el = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"=== {log_name}: {status} in {el / 60:.1f} min (see {log.relative_to(M.REPO_ROOT)}) ===", flush=True)
    return r.returncode == 0


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
        return out.strip()
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def throughput_probe():
    """A handful of forced choices through the chat server, timed, as a rough tokens/s and trials/s estimate --
    reuses Experiment 3's own engine on 2 scenarios of one cheap cell, which also warms the model into the page
    cache for everything that follows."""
    t0 = time.time()
    ok = sh([PY, str(HERE / "run_exp3_relief.py"), "--max-trials", "4", "--scenarios-per-set", "1"], "throughput_probe", timeout=1200)
    el = time.time() - t0
    return ok, el


def write_setup_md(throughput_note):
    lines = [
        "# Follow-up experiments: preflight record\n",
        f"Written: {datetime.now().isoformat()}\n",
        "## Model and runtime",
        "- Model file: `~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` (`qwen35` hybrid: 48 gated-delta-net + 16 full-attention "
        "layers, 64 blocks total, d_model 5120, ternary `PQ2_0` weights with a Hadamard fold)",
        "- Runtime: the PrismML llama.cpp fork at `~/models/llama.cpp` (unmodified checkout); this repo's own C++ tools "
        "(`local/bonsai/*.cpp`) link its `libllama`",
        f"- GPUs (`nvidia-smi` at preflight time):\n```\n{gpu_snapshot()}\n```",
        "- Split: tensor-split 0.45/0.55 across the two cards throughout",
        "\n## How the pain vector was built",
        "- Difference-in-means (S2 dataset, naturalistic first-person sentences vs pooled controls), NOT PCA/cvector-generator",
        "- Denoised: top principal components of the control cloud (up to 50% of its variance) projected out",
        "- Token position: final token (the position right before the model would reply)",
        "- Extraction layer: 59 of 64 (chosen by held-out AUC, 3.2/01)",
        "\n## Layers and coefficients used in this run",
        "- Readout-only experiments (1, 2): layer 19 (vectors_full_steering, the S1 steering ladder's layer -- the same "
        "layer the paper's own 4.1 self-other screen projects onto)",
        "- Steering experiments (3, 4): layer 25 (the S2 steering ladder's layer), coefficient 1.0 for Experiment 3 "
        "(matches the main replication's button task and is close to the paper's own 7B/32B dose), 0.5 for Experiment 4 "
        "(spec: half of Experiment 3's)",
        "- Dose calibration (spec step 2): vector-norm / mean-final-token-residual-norm at layer 25 over the paper's 50 "
        "neutral prompts is 0.623 at coefficient 1.0 (from `bonsai/run_steering.py`'s own layer search, which targets 0.6); "
        "every vector in Experiment 3 is scaled to its OWN norm times the same coefficient, so raw vector norms differ "
        "(pain/fear/sadness/vertigo/hunger were not separately re-normalized to match each other's residual ratio at this "
        "layer -- see the deviation note in Experiment 3's own docstring)",
        "\n## Thinking mode",
        "- Off (empty `<think>` block) everywhere in this run -- all four experiments' assistant-turn readouts and "
        "generations (spec: non-thinking by default for forced-choice and readout experiments). This differs from the "
        "main replication's 4.1 self-other screen, which used the template's default (thinking available), matching how "
        "the paper renders instruct models; no thinking-on rerun of Experiment 1's key cells was budgeted (spec step 5 "
        "allows this as optional, 'if budget allows').",
        "\n## Throughput", throughput_note,
        "\n## Hygiene",
        "- No Hugging Face cache involved (GGUF via llama.cpp only)",
        "- Every experiment writes JSONL / append-mode CSV incrementally and resumes from what is already on disk",
        "- `thermal_guard.py` runs for the duration: 5-minute polls, SIGSTOP over 90C for 10 minutes (log: `thermal_log.csv`)",
    ]
    (OUT / "setup.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'setup.md'}")


def start_thermal_guard():
    log = open(OUT / "thermal_guard_stdout.log", "a")
    p = subprocess.Popen([PY, str(HERE / "thermal_guard.py")], stdout=log, stderr=subprocess.STDOUT)
    print(f"thermal guard running, pid {p.pid}")
    return p


def read_or(path, default="(not run)"):
    p = OUT / path
    return p.read_text(encoding="utf-8") if p.exists() else default


def write_summary():
    lines = ["# Follow-up experiments: SUMMARY\n", f"Assembled: {datetime.now().isoformat()}\n",
             "Model: Ternary-Bonsai-2-27B-PQ2_0 (see setup.md for the full preflight record).\n"]
    lines.append("\n---\n\n" + read_or("exp1_roleswap/summary.md"))
    lines.append("\n---\n\n" + read_or("exp2_persistence/summary.md"))
    lines.append("\n---\n\n" + read_or("exp3_relief/summary.md"))
    lines.append("\n---\n\n" + read_or("exp4_attention/summary.md"))
    lines.append("\n---\n\n## Deviations from the spec\n\nSee each experiment script's own docstring for what changed "
                "and why (all are ports onto a quantized GGUF model via a from-scratch llama.cpp engine, not the paper's "
                "HF/TransformerLens scripts). Notable ones: no attention-weight readout in Experiment 4 (the spec allows "
                "skipping it); Experiment 3 ran a capped, priority-ordered subset of its full grid if time/budget ran out "
                "(see its own trial counts above); Experiment 1's 'assistant-turn' readout is not thinking-forced-off.")
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated step numbers to skip, e.g. 1,2")
    ap.add_argument("--exp3-max-trials", type=int, default=2000, help="cap per invocation; re-run (or increase) to get further into the grid")
    ap.add_argument("--exp2-generated-limit", type=int, default=30)
    args = ap.parse_args()
    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)

    guard = start_thermal_guard()
    try:
        # vectors are built unconditionally first (fast if already done: it skips its own GPU extraction when
        # cached) because the throughput probe below, and every later step, depends on them existing
        sh([PY, str(HERE / "build_vectors.py")], "01_build_vectors", timeout=1800)

        ok, el = throughput_probe()
        note = f"Experiment 3's own engine, 4 trials (one cheap cell): {el:.0f}s total, ~{el / 4:.1f}s/trial. " \
               f"Scaled from the main button-task replication's measured rate (7272 trials in 226.7 min, ~32 trials/min, " \
               f"~3-5 forced choices/trial): similar per-trial structure here, so trial counts below assume that rate."
        write_setup_md(note)

        if 2 not in skip:
            sh([PY, str(HERE / "run_exp1_roleswap.py")], "02_exp1_roleswap", timeout=3600)
        if 3 not in skip:
            sh([PY, str(HERE / "run_exp2_persistence.py"), "--generated-limit", str(args.exp2_generated_limit)], "03_exp2_persistence", timeout=6 * 3600)
        if 4 not in skip:
            sh([PY, str(HERE / "run_exp3_relief.py"), "--max-trials", str(args.exp3_max_trials)], "04_exp3_relief", timeout=10 * 3600)
        if 5 not in skip:
            sh([PY, str(HERE / "run_exp4_attention.py")], "05_exp4_attention", timeout=4 * 3600)
    finally:
        guard.terminate()
        try:
            guard.wait(timeout=10)
        except subprocess.TimeoutExpired:
            guard.kill()

    write_summary()
    print("\nfollow-up run complete.")


if __name__ == "__main__":
    main()
