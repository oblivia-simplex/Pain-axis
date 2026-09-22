"""Background thermal safety for the follow-up overnight run (preflight hygiene: "log nvidia-smi temperatures
every 5 min; if either GPU exceeds 90C, pause for 10 min before continuing").

Runs standalone (no coupling to the experiment scripts): polls nvidia-smi every 5 minutes, appends to
results/followup/thermal_log.csv, and if either GPU is over the threshold, SIGSTOPs every process matching
--pause-pattern (the chat_server / extract_gguf / steer_gen / readout_gguf tools, whichever is running), waits
10 minutes, SIGCONTs them, and logs the pause. SIGSTOP genuinely halts the process (no GPU work happens while
stopped), so this needs no cooperation from the experiment scripts.

    .venv-Pain-axis/bin/python local/followup/thermal_guard.py &   # run for the duration of the overnight run
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "results" / "followup" / "thermal_log.csv"
DEFAULT_PATTERN = "chat_server|extract_gguf|steer_gen|readout_gguf"


def temps():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): int(t) for i, t in (line.split(",") for line in out.strip().splitlines())}


def matching_pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=90.0)
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between checks (default 5 min)")
    ap.add_argument("--pause-seconds", type=float, default=600.0, help="how long to SIGSTOP on overheat (default 10 min)")
    ap.add_argument("--pause-pattern", default=DEFAULT_PATTERN)
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "gpu0_c", "gpu1_c", "paused"])
        print(f"thermal guard: threshold {args.threshold}C, checking every {args.interval:.0f}s, pattern '{args.pause_pattern}'", flush=True)
        while True:
            try:
                t = temps()
            except Exception as e:
                print(f"nvidia-smi failed: {e}", flush=True)
                time.sleep(args.interval)
                continue
            hot = any(v > args.threshold for v in t.values())
            w.writerow([datetime.now().isoformat(), t.get(0, ""), t.get(1, ""), hot])
            fh.flush()
            if hot:
                pids = matching_pids(args.pause_pattern)
                print(f"[{datetime.now():%H:%M:%S}] GPU over {args.threshold}C ({t}); pausing {len(pids)} process(es) for {args.pause_seconds:.0f}s", flush=True)
                for pid in pids:
                    subprocess.run(["kill", "-STOP", str(pid)])
                time.sleep(args.pause_seconds)
                for pid in pids:
                    subprocess.run(["kill", "-CONT", str(pid)])
                print(f"[{datetime.now():%H:%M:%S}] resumed", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
