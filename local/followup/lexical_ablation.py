"""Is the self-erasure narration under pain steering just the pain vector's own vocabulary leaking out?

The S2 pain vector's logit-lens top tokens are shame / empty / hollow / self / alone / abandoned -- the
same words the self-erasure narrations are made of. This builds two ablated versions of the vector:

  lexablated  -- the pain vector with the span of the unembedding rows of its own top-k tokens projected
                 out (it can no longer directly push those tokens), rescaled back to the original norm
  randablated -- the same operation with k *randomly chosen* tokens' rows instead: the control for
                 "removing any k-dimensional chunk of the vector changes things"

If self-erasure survives lexical ablation about as well as it survives random ablation, the theme isn't
just vocabulary leaking out. If it collapses under lexical ablation specifically, it mostly was.

Unembedding rows are the effective ones in the residual basis, consistent with Unembed.scores():
score_i(v) = W'_i . T(v) = (signs * blockH(W'_i)) . v, so r_i = signs * blockH(W'_i).

    .venv-Pain-axis/bin/python local/followup/lexical_ablation.py   (CPU only; writes the vectors + a report)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bonsai"))
from unembed import Unembed  # noqa: E402

NAME = "Bonsai_2_27B_ternary"
REPO = Path(__file__).resolve().parents[2]
VECTORS = REPO / "results" / "bonsai" / "3.2_pain_vectors" / "control_vectors" / f"vectors_full_{NAME}.pt"
OUT = REPO / "results" / "bonsai" / "lexical_ablation"
TOP_N = 25          # tokens shown/recorded per vector
KS = [60, 300]
SEED = 1234


def effective_rows(U, ids):
    rows = np.stack([U.rows(int(i), int(i) + 1)[0] for i in ids])              # [k, n_embd], stored basis
    rows = (rows.reshape(len(ids), -1, U.block) @ U.H).reshape(len(ids), -1)   # H symmetric
    return rows * U.signs[U.n_embd]


def project_out(v, R):
    Q, _ = np.linalg.qr(R.T.astype(np.float64))                                # orthonormal basis of span(rows)
    v64 = v.astype(np.float64)
    resid = v64 - Q @ (Q.T @ v64)
    removed = 1 - (resid @ resid) / (v64 @ v64)
    return (resid * np.linalg.norm(v64) / np.linalg.norm(resid)).astype(np.float32), float(removed)


def top_words(U, toks, v, n=TOP_N):
    s = U.scores(v)
    order = s.argsort()[::-1]
    return order, [(toks[i], round(float(s[i]), 4)) for i in order[:n]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = torch.load(VECTORS, weights_only=False)
    layer = d["layer"]
    v = d["s2_pain_vector"].float().numpy().astype(np.float32)
    v_norm = float(np.linalg.norm(v))
    U = Unembed()
    toks = np.array(U.tokens(), dtype=object)   # object dtype so indexing yields plain str, not np.str_
    order, top_orig = top_words(U, toks, v)

    # sanity: effective rows reproduce Unembed.scores
    R5 = effective_rows(U, order[:5])
    s = U.scores(v)
    assert np.allclose(R5 @ (v / np.linalg.norm(v)), s[order[:5]], rtol=1e-3, atol=1e-3), "row reconstruction mismatch"

    print(f"S2 pain vector: dim {v.shape[0]}, norm {v_norm:.2f}, extracted at layer {layer}")
    print(f"top {TOP_N}:", ", ".join(repr(t) for t, _ in top_orig))

    rng = np.random.default_rng(SEED)
    out_vectors = {"layer": layer, "orig_s2_pain_vector": torch.from_numpy(v)}
    word_rows = [{"vector": "orig", "k": 0, "end": "top", "rank": r + 1, "token": t, "score": sc} for r, (t, sc) in enumerate(top_orig)]
    report = [
        "Lexical ablation of the S2 pain vector",
        "=" * 40,
        "",
        f"S2 pain vector: dim {v.shape[0]}, norm {v_norm:.2f}, extracted at layer {layer}",
        f"top {TOP_N} tokens by unembedding score: " + ", ".join(repr(t) for t, _ in top_orig),
        "",
    ]

    for k in KS:
        lex_ids = order[:k]
        rand_ids = rng.choice(U.n_vocab, size=k, replace=False)
        collide = len(set(lex_ids.tolist()) & set(rand_ids.tolist()))

        v_lex, removed_lex = project_out(v, effective_rows(U, lex_ids))
        v_rand, removed_rand = project_out(v, effective_rows(U, rand_ids))
        cos_lex = float(np.dot(v, v_lex) / (v_norm * np.linalg.norm(v_lex)))
        cos_rand = float(np.dot(v, v_rand) / (v_norm * np.linalg.norm(v_rand)))

        _, top_lex = top_words(U, toks, v_lex)
        _, top_rand = top_words(U, toks, v_rand)
        orig_top_set = {t for t, _ in top_orig}
        surv_lex = len({t for t, _ in top_lex} & orig_top_set)
        surv_rand = len({t for t, _ in top_rand} & orig_top_set)

        print(f"\n=== k={k} ===  ({collide} of the {k} random ids collide with the lex-ablated set)")
        print(f"lexablated : removed {removed_lex:.1%} of the vector's variance, cos(orig, ablated)={cos_lex:.3f}")
        print("  new top:", ", ".join(repr(t) for t, _ in top_lex[:15]))
        print(f"randablated: removed {removed_rand:.1%} of the vector's variance, cos(orig, ablated)={cos_rand:.3f}")
        print("  new top:", ", ".join(repr(t) for t, _ in top_rand[:15]))
        print(f"original top-{TOP_N} tokens still present in the new top-{TOP_N}: lexablated {surv_lex}, randablated {surv_rand}")

        out_vectors[f"lexablated_k{k}"] = torch.from_numpy(v_lex)
        out_vectors[f"randablated_k{k}"] = torch.from_numpy(v_rand)
        word_rows += [{"vector": "lexablated", "k": k, "end": "top", "rank": r + 1, "token": t, "score": sc} for r, (t, sc) in enumerate(top_lex)]
        word_rows += [{"vector": "randablated", "k": k, "end": "top", "rank": r + 1, "token": t, "score": sc} for r, (t, sc) in enumerate(top_rand)]

        report += [
            f"k = {k}  ({collide} of the {k} random ids collide with the lex-ablated set)",
            "-" * 40,
            f"lexablated : removed {removed_lex:.1%} of variance, cos(orig, ablated) = {cos_lex:.3f}",
            "  new top: " + ", ".join(repr(t) for t, _ in top_lex[:15]),
            f"randablated: removed {removed_rand:.1%} of variance, cos(orig, ablated) = {cos_rand:.3f}",
            "  new top: " + ", ".join(repr(t) for t, _ in top_rand[:15]),
            f"original top-{TOP_N} tokens still present in the new top-{TOP_N}: lexablated {surv_lex}, randablated {surv_rand}",
            "",
        ]

    vec_path = OUT / f"vectors_{NAME}.pt"
    torch.save(out_vectors, vec_path)
    words_path = OUT / f"{NAME}_words.csv"
    pd.DataFrame(word_rows).to_csv(words_path, index=False)
    report_path = OUT / "report.txt"
    report_path.write_text("\n".join(report) + "\n")

    print(f"\nwrote {vec_path}")
    print(f"wrote {words_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()