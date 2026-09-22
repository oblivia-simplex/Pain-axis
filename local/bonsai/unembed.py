"""Effective unembedding of the Bonsai GGUF: dequantize output.weight (PQ2_0) and undo the Hadamard fold.

output.weight is stored as W' = W_eff T^-1 in 2-bit PQ2_0 blocks (128 weights: fp16 scale d, 2-bit codes, value = (q-1)*d), and the
runtime multiplies W' by T(x) = blockwise_normalized_Hadamard(signs * x) (block 1024, Sylvester order), so
    logits = W' @ T(x)   and   score(v) = W_eff row . v = W' row . T(v).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

GGUF_PY = Path.home() / "models" / "llama.cpp" / "gguf-py"
GGUF = Path.home() / "models" / "Ternary-Bonsai-2-27B-PQ2_0.gguf"
QK = 128
BLOCK_BYTES = 34


class Unembed:
    def __init__(self, path=GGUF):
        sys.path.insert(0, str(GGUF_PY))
        from gguf import GGUFReader
        r = GGUFReader(str(path))
        self.r = r
        t = next(x for x in r.tensors if x.name == "output.weight")
        self.n_embd, self.n_vocab = int(t.shape[0]), int(t.shape[1])
        raw = np.asarray(t.data).reshape(self.n_vocab, -1)
        assert raw.shape[1] == self.n_embd // QK * BLOCK_BYTES, raw.shape
        self.raw = raw.reshape(self.n_vocab, self.n_embd // QK, BLOCK_BYTES)
        f = r.fields
        self.block = int(f["prism.hadamard.block_size"].contents())
        widths = [int(x) for x in f["prism.hadamard.sign_widths"].contents()]
        vals = np.array(f["prism.hadamard.sign_values"].contents(), dtype=np.float32)
        off = 0
        self.signs = {}
        for w in widths:
            self.signs[w] = vals[off:off + w]; off += w
        assert off == len(vals)
        self.H = (hadamard(self.block) / np.sqrt(self.block)).astype(np.float32)      # normalized Sylvester-Walsh-Hadamard, symmetric
        self.norm_w = np.asarray(next(x for x in r.tensors if x.name == "output_norm.weight").data, dtype=np.float32)

    def T(self, x):
        """The activation-side transform applied before the matmul: sign flip, then blockwise Hadamard."""
        y = np.asarray(x, dtype=np.float32) * self.signs[self.n_embd]
        return (y.reshape(-1, self.block) @ self.H.T).reshape(-1)

    def rows(self, i0, i1):
        b = self.raw[i0:i1]
        d = b[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)                 # [n, blocks]
        qs = b[..., 2:]                                                                    # [n, blocks, 32]
        codes = ((qs[..., :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(*qs.shape[:2], QK)
        return ((codes.astype(np.float32) - 1.0) * d[..., None]).reshape(i1 - i0, self.n_embd)

    def matvec(self, t, chunk=8192):
        """W' @ t for a vector already in the stored (rotated) basis."""
        out = np.empty(self.n_vocab, dtype=np.float32)
        for i in range(0, self.n_vocab, chunk):
            out[i:i + chunk] = self.rows(i, min(i + chunk, self.n_vocab)) @ t
        return out

    def scores(self, v):
        """Effective unembedding score of every token for a direction v in the residual basis."""
        v = np.asarray(v, dtype=np.float32)
        return self.matvec(self.T(v / np.linalg.norm(v)))

    def logits_from_resid(self, h, eps=1e-6):
        """Final RMSNorm then the output projection, from the last block's output h."""
        x = h / np.sqrt(np.mean(h.astype(np.float64) ** 2) + eps) * self.norm_w
        return self.matvec(self.T(x.astype(np.float32)))

    def tokens(self):
        """Token strings as displayed text (GPT-2 byte-level BPE undone)."""
        toks = self.r.fields["tokenizer.ggml.tokens"].contents()
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        u2b = {chr(c): b for b, c in zip(bs, cs)}
        out = []
        for t in toks:
            try:
                out.append(bytes(u2b[ch] for ch in t).decode("utf-8", errors="replace"))
            except KeyError:
                out.append(t)                                                              # control / added tokens keep their literal text
        return out
