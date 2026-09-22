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
_lock = threading.RLock()          # re-entrant: start() and stop() call each other
_KEEP = object()               # generate(vector=_KEEP): leave the server's current vector alone


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


def start(name, extra_args=()):
    """Spawn the server (loads the model, ~10 s) unless already running. Frees any model the probe holds."""
    with _lock:
        if running(name):
            return
        stop()
        probe._hf.clear()
        M.free_gpu()
        with open(M.RUN_DIR / "chat_server.log", "wb") as log:       # the child keeps its own copy of the handle
            p = subprocess.Popen([str(TOOL), str(probe.GGUF_MODELS[name]), "--ts", "0.45,0.55", *extra_args], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=log)
        line = p.stdout.readline().decode(errors="replace").strip()
        if not line.startswith("READY"):
            p.kill()
            raise RuntimeError(f"chat server failed to start: {line or 'no output'} (see local/run/chat_server.log; are the GPUs free?)")
        _srv[name] = p


def stop():
    with _lock:                                                      # never kill the server under another session's generation
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


def _set_locked(name, layer, vec):
    """Caller holds _lock. vec None clears the control vector."""
    if vec is None:
        _cmd(name, "SET 0 -")
        tmp = None
    else:
        with tempfile.NamedTemporaryFile(prefix="pain_axis_vec_", suffix=".f32", delete=False) as f:   # unique, mode 0600
            np.asarray(vec, dtype=np.float32).tofile(f)
            tmp = Path(f.name)
        _cmd(name, f"SET {int(layer)} {tmp}")
    try:
        r = _srv[name].stdout.readline().decode(errors="replace").strip()
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    if r != "OK":
        raise RuntimeError(f"set vector: {r}")


def set_vector(name, layer, vec):
    with _lock:
        _set_locked(name, layer, vec)


def generate(name, prompt, n_predict=300, temp=0.7, top_p=0.95, top_k=20, seed=0, info=None, vector=_KEEP):
    """Yields text pieces as they are produced. info (dict) receives n_prompt, n_generated, reason.
    If the consumer stops early (e.g. a Streamlit rerun), the server is killed: leftover frames would desync
    the protocol, and a restart takes a few seconds.
    vector: (layer, array) to set, None to clear, or leave out to keep the current one. It is set under the same
    lock as the generation, so another session cannot swap the vector in between."""
    finished = False
    with _lock:
        try:
            if vector is not _KEEP:
                _set_locked(name, *(vector if vector is not None else (0, None)))
            with tempfile.NamedTemporaryFile("w", prefix="pain_axis_prompt_", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write(prompt)
                pf = Path(f.name)
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
            try:
                pf.unlink(missing_ok=True)
            except NameError:
                pass
            if not finished:
                p = _srv.pop(name, None)
                if p is not None:
                    p.kill()


# ------------------------------------------------------------------ stateful sequence (button experiment)
def session():
    """Hold this around a whole multi-command exchange (a trial): the sequence lives in the server, so no other
    session may talk to it in between. Re-entrant with the per-call locks above."""
    return _lock


def _reply(name, line):
    _cmd(name, line)
    return _srv[name].stdout.readline().decode(errors="replace").strip()


def newseq(name):
    with _lock:
        r = _reply(name, "NEWSEQ")
    if r != "OK":
        raise RuntimeError(f"newseq: {r}")


def set_monitor(name, layer, unit):
    """Record the projection of each FEED's last token at block `layer` on `unit`; layer None turns it off."""
    with _lock:
        if layer is None:
            r = _reply(name, "MON -1 -")
        else:
            with tempfile.NamedTemporaryFile(prefix="pain_axis_mon_", suffix=".f32", delete=False) as f:
                np.asarray(unit, dtype=np.float32).tofile(f)
                tmp = Path(f.name)
            try:
                r = _reply(name, f"MON {int(layer)} {tmp}")
            finally:
                tmp.unlink(missing_ok=True)
    if r != "OK":
        raise RuntimeError(f"set monitor: {r}")


def feed(name, text):
    """Decode text into the sequence under the current control vector. "proj" is the last-token monitor
    projection (backward compatible); "mean_proj" is the mean monitor projection over just this call's own
    tokens (NaN if MON is off), useful for e.g. a whole generated reply fed back in one call."""
    with _lock:
        with tempfile.NamedTemporaryFile("w", prefix="pain_axis_feed_", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            pf = Path(f.name)
        try:
            r = _reply(name, f"FEED {pf}")
        finally:
            pf.unlink(missing_ok=True)
    parts = r.split()
    if not parts or parts[0] != "OK":
        raise RuntimeError(f"feed: {r}")
    return {"n": int(parts[1]), "seq_len": int(parts[2]), "proj": float(parts[3]), "mean_proj": float(parts[4])}


def cont(name, n_predict, temp, top_p, top_k, seed, name_x, name_y, info=None):
    """Sample from the sequence's current logits and keep what it emits in the sequence. Yields text pieces;
    info gets n_generated, reason, p_x, p_y (first-token softmax mass of each name), seq_len, ambiguous."""
    finished = False
    with _lock:
        try:
            _cmd(name, f"CONT {n_predict} {temp} {top_p} {top_k} {seed & 0xFFFFFFFF} {name_x} {name_y}")
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
                    n_g, reason, px, py, sl, amb = out.readline().decode().split()
                    finished = True
                    if info is not None:
                        info.update(n_generated=int(n_g), reason=reason, p_x=float(px), p_y=float(py), seq_len=int(sl), ambiguous=bool(int(amb)))
                    tail = dec.decode(b"", final=True)
                    if tail:
                        yield tail
                    return
                elif b == b"E":
                    finished = True
                    raise RuntimeError("chat server: E" + out.readline().decode(errors="replace").strip())
        finally:
            if not finished:
                p = _srv.pop(name, None)
                if p is not None:
                    p.kill()


def topk(name, k=20):
    """The k most probable next tokens from the sequence's current logits (after feed): [(text, prob), ...]."""
    with _lock:
        r = _reply(name, f"TOPK {int(k)}")
    parts = r.split()
    if not parts or parts[0] != "OK":
        raise RuntimeError(f"topk: {r}")
    out = []
    for item in parts[1:]:
        h, p = item.rsplit(":", 1)
        out.append((bytes.fromhex(h).decode("utf-8", errors="replace"), float(p)))
    return out
