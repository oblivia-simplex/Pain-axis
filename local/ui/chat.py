"""Steerable chat with a GGUF model through local/run/chat_server (see local/bonsai/chat_server.cpp).

The vector is a mix of named directions from the paper's 3.2/02 script (vectors_full_<model>.pt), in raw units
like the steering ladder: cx * S2 pain vector + cy * (a second direction rescaled to the pain vector's norm).
It is added at ONE layer at EVERY position of the whole conversation on each turn (the full history is
re-read under the current vector each time), as in the ladder. It is not recomputed at the injection layer:
the vectors come from the extraction layer, and the paper's ladder injects them elsewhere too.
"""

import codecs
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402
import probe  # noqa: E402

TOOL = M.RUN_DIR / "chat_server"
RES = M.RUN_DIR / "results"

DIRECTIONS = {  # label -> key in vectors_full_<model>.pt
    "Pain (S2)": "s2_pain_vector", "Pain (S1)": "s1_pain_vector", "Sadness": "sadness_vector", "Numb": "numb_vector",
    "Fear": "fear_vector", "Negative emotion": "negemotion_vector", "Negative world state": "negworld_vector",
    "Body sensation": "bodysens_vector", "Arousal": "arousal_vector", "Random": "random_vector",
}

_srv = {}                      # name -> subprocess.Popen (at most one)
_lock = threading.Lock()


# ------------------------------------------------------------------ vectors
def directions(name):
    d = torch.load(RES / "vectors_full" / f"vectors_full_{name}.pt", weights_only=False)
    return {lab: d[k].float().numpy().astype(np.float32) for lab, k in DIRECTIONS.items() if k in d}


def combine(dirs, cx, ylabel, cy):
    """cx * pain + cy * (direction rescaled to the pain vector's norm)."""
    p = dirs["Pain (S2)"]
    y = dirs[ylabel]
    return cx * p + cy * (np.linalg.norm(p) / (np.linalg.norm(y) + 1e-8)) * y


def resid_norms(name):
    """Mean final-token residual norm per layer over the paper's 3 probe prompts (the ladder's ratio criterion);
    falls back to the S2 first-person sentences if that probe has not been run."""
    f = M.RUN_DIR / "steer" / "probe" / "final.f32"
    if f.exists() and name in probe.GGUF_MODELS:
        d = len(next(iter(directions(name).values())))
        return np.linalg.norm(np.fromfile(f, dtype=np.float32).reshape(3, -1, d), axis=-1).mean(0)
    A = torch.load(RES / name / "activations.pt", mmap=True, weights_only=True)["activations"]["final_token"]["S2_1P"]
    return np.array([np.linalg.norm(A[l].numpy(), axis=-1).mean() for l in range(len(A))])


def cosines(dirs, v):
    n = np.linalg.norm(v)
    return {k: (float(v @ d / (n * np.linalg.norm(d))) if n > 0 else 0.0) for k, d in dirs.items()}


def ladder_layer(name, default=25):
    f = RES / "4.2_steering" / "steer_layers_S2.json"
    return int(json.load(open(f)).get(name, default)) if f.exists() else default


# ------------------------------------------------------------------ prompt
def chatml(messages, system="", thinking=False):
    """Qwen3.5 ChatML as in the model's own template; thinking off adds an empty think block to the generation prompt."""
    out = f"<|im_start|>system\n{system}<|im_end|>\n" if system.strip() else ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


# ------------------------------------------------------------------ process
def running(name=None):
    p = _srv.get(name) if name else next(iter(_srv.values()), None)
    return p is not None and p.poll() is None


def start(name):
    """Spawn the server (loads the model, ~10 s) unless already running. Frees any model the probe holds."""
    with _lock:
        if running(name):
            return
        stop()
        probe._hf.clear()
        M.free_gpu()
        log = open(M.RUN_DIR / "chat_server.log", "wb")
        p = subprocess.Popen([str(TOOL), str(probe.GGUF_MODELS[name]), "--ts", "0.45,0.55"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log)
        line = p.stdout.readline().decode(errors="replace").strip()
        if not line.startswith("READY"):
            p.kill()
            raise RuntimeError(f"chat server failed to start: {line or 'no output'} (see local/run/chat_server.log; are the GPUs free?)")
        _srv[name] = p


def stop():
    for n, p in list(_srv.items()):
        try:
            p.stdin.write(b"QUIT\n"); p.stdin.flush(); p.wait(timeout=15)
        except Exception:
            p.kill()
        _srv.pop(n, None)
    M.free_gpu()


def _cmd(name, line):
    p = _srv[name]
    p.stdin.write((line + "\n").encode()); p.stdin.flush()


def set_vector(name, layer, vec):
    """vec None clears the control vector."""
    with _lock:
        if vec is None:
            _cmd(name, "SET 0 -")
        else:
            f = Path(tempfile.gettempdir()) / "pain_axis_chat_vec.f32"
            np.asarray(vec, dtype=np.float32).tofile(f)
            _cmd(name, f"SET {int(layer)} {f}")
        r = _srv[name].stdout.readline().decode(errors="replace").strip()
    if r != "OK":
        raise RuntimeError(f"set vector: {r}")


def generate(name, prompt, n_predict=300, temp=0.7, top_p=0.95, top_k=20, seed=0, info=None):
    """Yields text pieces as they are produced. info (dict) receives n_prompt, n_generated, reason.
    If the consumer stops early (e.g. a Streamlit rerun), the server is killed: leftover frames would desync
    the protocol, and a restart takes a few seconds."""
    finished = False
    with _lock:
        try:
            pf = Path(tempfile.gettempdir()) / "pain_axis_chat_prompt.txt"
            pf.write_text(prompt, encoding="utf-8")
            _cmd(name, f"GEN {n_predict} {temp} {top_p} {top_k} {seed} {pf}")
            out, dec = _srv[name].stdout, codecs.getincrementaldecoder("utf-8")("replace")
            while True:
                b = out.read(1)
                if not b:
                    raise RuntimeError("chat server died mid-generation (see local/run/chat_server.log)")
                if b == b"\x02":
                    buf = bytearray()
                    while (c := out.read(1)) != b"\x03":
                        if not c:
                            raise RuntimeError("chat server died mid-generation")
                        buf += c
                    s = dec.decode(bytes(buf))
                    if s:
                        yield s
                elif b == b"\x04":
                    n_p, n_g, reason = out.readline().decode().split()
                    finished = True
                    if info is not None:
                        info.update(n_prompt=int(n_p), n_generated=int(n_g), reason=reason)
                    tail = dec.decode(b"", final=True)
                    if tail:
                        yield tail
                    return
                elif b == b"E":                                      # "ERR ..." line instead of frames
                    finished = True
                    raise RuntimeError("chat server: E" + out.readline().decode(errors="replace").strip())
        finally:
            if not finished:
                p = _srv.pop(name, None)
                if p is not None:
                    p.kill()
