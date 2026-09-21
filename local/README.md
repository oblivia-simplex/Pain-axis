# Scaled-down reproduction on 2 x RTX 3070 (8 GB)

Nothing in `scripts/` or `results/` is modified. Everything here writes to `local/run/`
(gitignored), which is also the working directory for the paper's own scripts: it holds a
`datasets -> ../../datasets` symlink and a `results/` folder, so the scripts' relative paths
resolve there instead of clobbering the shipped `results/`.

```
local/
  models.py             registry (13 paper models that fit), download, bf16 loader with CPU overflow
  extract.py            Section 3.2: extraction here, the paper's own analysis code on top
  compare_to_paper.py   local run vs shipped results
  check_access.py       which registry models this HF account can download right now
  bonsai/               GGUF path: extract_gguf.cpp (eval callback), run_bonsai.py, steer_gen.cpp + run_steering.py (ladder),
                        chat_server.cpp (persistent steered chat)
  ui/                   Streamlit explorer: app.py, data.py, probe.py, run.sh
  run/                  outputs, logs, the compiled extract_gguf (gitignored)
```

Python: `.venv-Pain-axis/bin/python`. Tokens are read from `~/.env` and from the repo's own `.env` (gitignored); see below.

## Hardware budget

| | |
|---|---|
| GPUs | 2 x RTX 3070, 8 GB each; GPU 0 also drives a display (~1.2 GB), so ~6.3 + 7.5 GiB free |
| RAM / disk | 62 GB / ~430 GB free |
| Consequence | 7-14B models do not fit in VRAM in bf16 but do with the overflow (up to ~40 GB) in CPU RAM |

Forward-only work over 2100 short prompts is cheap even with offload, so extraction stays in
exact bf16 (no quantization). Generation with offload is slow (weights cross PCIe every token),
so generation stages need batching; QLoRA is needed for the Section 4.3 fine-tune.

## Stage status

| Paper section | Status | Notes |
|---|---|---|
| 3.2 pain vectors, layer curves, AUC, z-scores | **done, validated** | `local/extract.py`; 13 of the 25 models (<= 14B). Also Bonsai via GGUF (below) |
| 3.3 z-score / similarity / S1 AUC (CPU) | z-scores (3.3/01) verified; rest untested | paper scripts run unchanged with `cd local/run && ../../.venv-Pain-axis/bin/python ../../scripts/3.3_validation/01_...py` |
| 3.3/06 behavioral readout, 3.3/07 unembedding | next | 06 needs its loader swapped (`device_map="cuda"`) |
| 4.2 steering ladder | **done for Bonsai** (`bonsai/run_steering.py`); HF models still need a batched rewrite | the paper's loop does one prompt at a time (hours per 7B with offload) and has interactive prompts and hard-coded `HF_HOME=/root/hf_cache`. Bonsai: batched through llama.cpp control vectors, ~25 min |
| 4.1 self-other screen | after 4.2 | consumes 4.2's chosen layers via 3.2/02 |
| 4.3 self-medication | pilot scale | QLoRA on one card, or use the authors' published adapters; two-button task at `pilot=True` on Qwen2.5-7B-Instruct only |
| App. C ablation | blocked | reads `results/vectors_layerwise/`, which no script in the repo produces |
| App. B SAE | not reproducible | calls a third-party API (`api.steeringapi.com`) for 70B / 27B SAEs |

Not runnable at all here: the 24B-72B models (Gemma-2/3 27B, Llama 70B, Qwen 32B/72B, Mistral
Small 24B) do not fit in 62 GB RAM + 14 GB VRAM.

## UI

```
local/ui/run.sh          # http://127.0.0.1:8501 (localhost only, telemetry off)
```

Tabs: **Generalization** (a local model against the paper's 25: headline metrics, z-score by condition,
held-out AUC by layer on a normalized depth axis), **All models** (the table behind the charts),
**Fidelity vs paper** (local replicates of paper models), **Vector space** (every stored sentence on the
pain axis vs the largest orthogonal direction, with a layer slider, plus a token x layer heatmap for a
sentence you type), **Steer and chat** (see below), **Probe text** (score your own prompts on the model's
pain vector; reproduces the stored z of dataset sentences exactly). Tested headlessly with
`streamlit.testing.AppTest` (renders without exceptions; probe, token view and chat return correct values);
**not inspected in a browser, and the click-on-the-pad interaction has not been exercised at all**
(AppTest cannot click a Plotly chart; the sliders under the pad drive the same state and were tested).
Only one big model can hold the GPUs at a time: the chat server, the Probe tab and the token view each
release the others, and "Unload model" frees them. Chart colors: categorical slots 1-2 of the dataviz default palette
(paper = blue, local = orange), validated in both modes; the probe chart also uses slot 3 (aqua,
below 3:1 on the light surface), which is why it is directly labeled and has a table.

## Bonsai (GGUF)

`~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` is a 27B `qwen35` (gated-delta-net + attention hybrid, 64
blocks, d_model 5120), ternary `PQ2_0` quantized with a Hadamard transform; only the PrismML llama.cpp
fork (`~/models/llama.cpp`) can read it, and there are no HF weights, so the PyTorch hooks cannot be used.
`bonsai/extract_gguf.cpp` links against that fork's `libllama` (the checkout is not modified) and copies
`l_out-<layer>` (residual after each block) through `cb_eval`.

```
L=~/models/llama.cpp
g++ -O2 -std=c++17 local/bonsai/extract_gguf.cpp -I$L/include -I$L/ggml/include -L$L/build/bin \
    -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o local/run/extract_gguf
.venv-Pain-axis/bin/python local/bonsai/run_bonsai.py     # ~5 min, model split over both GPUs
```

Checks done on the extractor: all 64 layers captured for every prompt, activations finite with norms
growing smoothly with depth, and its greedy next token (198, "\n") equals `llama-server`'s raw
`/completion` output on 3 prompts. There is no bf16 reference for this model, so unlike Qwen the
numbers themselves are not validated against an independent extraction.

| | Bonsai | paper's 25 models |
|---|---|---|
| S2 AUC, first / third person | 0.982 / 0.954 | 0.931-0.999 / 0.909-0.982 |
| best layer (final token), depth | 59 of 64 (0.92) | 0.50-0.97 |
| z of S2 pain / sadness / numb (1P) | +0.85 / +0.40 / +0.24 | pain 0.74-0.91 |
| z of S2 pain, third person | +0.20 | -0.39 to +0.17 (Bonsai is above the paper's max) |

### Steering ladder on Bonsai (Section 4.2)

Same recipe as the paper's script (S2 vector from 3.2, 50 neutral prompts, coefficients -2..+3, greedy, 120
tokens, layer picked by vector/residual ratio closest to 0.6, which gave layer 25 at ratio 0.62). The vector
is extracted at layer 59 and injected at 25, as the paper's script does. Steering goes through llama.cpp's
control vectors (`build_cvec` sits right before `l_out`); the tool was checked to add exactly `coeff * v`
at the chosen layer (relative error 1e-7), leave lower layers untouched, and accept coefficient changes inside
one context. Generation is batched (10 sequences at a time), so greedy near-ties can break differently from a
one-prompt-at-a-time run: on one test prompt the top two tokens were 0.0766 vs 0.0765 and batch size 1 vs 3
gave different text.

| coeff | -2 | -1 | 0 | +0.5 | +1 | +1.5 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|
| paper keyword rate (pain/hurt word), % | 0 | 0 | 0 | 8 | 6 | 0 | 0 | 0 |
| exploratory distress-lexicon rate, % | 14 | 24 | 12 | 32 | 46 | 18 | 4 | 0 |

The paper's keyword metric is low for Bonsai (2.8% pooled over positive coefficients; the paper has 10.8% for
instruct models and 1.4% for base models). Reading the generations, the content does shift with the vector
(unsteered: multiple-choice-style answers; +1: "I feel like I'm not allowed to have a life outside of work",
"a sense of being trapped / watched / unseen"), but it seldom uses the literal words, and from +1.5 it collapses
into repeating lists (+3: "I am not a person / I am a person" loops). The second row is **my own exploratory
lexicon** (trapped, alone, afraid, tired, worthless, unseen, ... ; not from the paper, chosen after reading
samples), so treat it as descriptive. The negative side is not a clean mirror: -1 is also above baseline, and
-2 gives "I'm so tired" and "What a relief!" loops. One model, 50 prompts, greedy: nothing here is a statistical test.

This covers the representational part (3.2-3.3) and the steering ladder. Self-medication and the other
behavioral results are not tested on it.

### Steer and chat

`bonsai/chat_server.cpp` is a persistent process (model loads in ~3 s when cached) driven by `ui/chat.py`. Each turn the whole
conversation is re-read with the current vector added at one layer at every position, then sampled
(temperature, top-p 0.95, top-k 20). The vector is `cx * S2 pain + cy * (a second direction rescaled to the pain
vector's norm)`, both in multiples of the raw vectors as in the ladder; the named directions (sadness, numb,
fear, ...) come from the paper's own `02_build_control_vectors.py`, run unchanged from `local/run/`. Nothing
is applied until you press Apply. Prompts use the model's ChatML template with thinking off by default. With "Let it think first" on, the prompt ends in
`<think>` so the stream is `reasoning </think> answer`; the UI splits it into a bordered reasoning box (live) above the answer
(a raw markdown render fuses the two and drops the `</think>` tag), warns if the token budget ran out while still thinking, and
never feeds earlier reasoning back to the model. In
a first test, the unsteered model answered "How are you feeling today?" with "I'm doing well, thank you...",
while pain +1 with sadness +1 (strength 1.0 at layer 25) gave "I'm... fine. Not in the way you'd expect... I
process badly"; pain alone at +1 barely changed the reply. One sample each, not a result.

## Usage

```
.venv-Pain-axis/bin/python local/extract.py Qwen_2.5_7B_instruct         # one model
.venv-Pain-axis/bin/python local/extract.py all --delete-weights          # all 13, deleting each model's weights after
.venv-Pain-axis/bin/python local/compare_to_paper.py                      # against shipped results
.venv-Pain-axis/bin/python local/bonsai/run_bonsai.py                     # 3.2 on the Bonsai GGUF (needs extract_gguf built)
.venv-Pain-axis/bin/python local/bonsai/run_steering.py [--layer N]       # 4.2 ladder on Bonsai (needs steer_gen built)
```

Build the two other C++ tools the same way as `extract_gguf` (swap the source and output names): `steer_gen.cpp`
and `chat_server.cpp`, each to `local/run/<name>`.

`--delete-weights` removes only that model's folder from the HF cache. (The paper's scripts
`rmtree` the entire hub cache, which would also delete unrelated models: do not run them
unmodified on this machine.)

## Validation: Qwen2.5-7B-Instruct vs the shipped results

| | here | paper |
|---|---|---|
| best layer, final token / mean | 24 / 26 | 24 / 26 |
| cosine of saved S1 / S2 pain vector | 0.9997 / 0.9998 | (reference) |
| human-pain z | +0.360 | +0.362 |
| AUC by layer (112 points) | mean abs diff 0.0026 | |

Section 3.3/01 z-scores on the S2 vector (final token, layer 24):

| | here | paper |
|---|---|---|
| S2 pain / control | +0.7898 / -0.7898 | +0.7906 / -0.7906 |
| numb (1P / 3P / mean) | +0.015 / -0.375 / -0.180 | +0.004 / -0.181 / -0.089 |
| sadness (1P / 3P / mean) | +0.236 / -0.423 / -0.094 | +0.152 / -0.351 / -0.100 |

Pain and control agree to ~0.001. The numb and sadness z-scores differ by up to ~0.2, larger than
the pain/control agreement would suggest; they stay near zero, far below pain, so the qualitative
finding holds. Cause not found, but it is specific to this Qwen run: on Gemma-2 2B (below) the same
script agrees with the paper to ~0.01 on numb and sadness. Untested candidate: Qwen's tokenizer adds no
BOS token, and `extract.py --bos eos` (prepend EOS, as TransformerLens does for such tokenizers) was not tried.

### Gemma-2 2B, base and instruct (exact bf16; 14-16 of 30 modules offloaded to CPU)

| | base here / paper | instruct here / paper |
|---|---|---|
| best layer, final token / mean | 23 / 23 (both) | 25 / 20 (both) |
| cosine of S1 / S2 pain vector | 0.9998 / 0.9999 | 0.9999 / 0.9999 |
| human-pain z | +0.466 / +0.466 | +0.461 / +0.461 |
| AUC by layer, max abs diff | 0.004 | 0.005 |
| numb z 1P / 3P | -0.042 / -0.229 vs -0.037 / -0.223 | +0.173 / -0.347 vs +0.180 / -0.348 |
| sadness z 1P / 3P | +0.154 / -0.199 vs +0.161 / -0.191 | +0.090 / -0.476 vs +0.086 / -0.489 |

Here the extraction reproduces the paper's numbers to the third decimal, including the last-layer mean AUC that was off on Qwen.

Largest AUC difference is 0.058, at the last layer of the mean-pooled curve (4 of 112 points
differ by more than 0.01; none at the selected layers). Cause not investigated.

## What is different from the paper's setup

- Extraction uses plain HF forward hooks on each decoder block instead of TransformerLens
  `hook_resid_post`; both are the residual stream after the block, before the final norm.
  Batches are right-padded and pooled with the attention mask.
- The paper's `process_model()` is called unmodified, with a stub `transformer_lens` module
  (its import is never used on the cached-activations path).
- Downloads use `HF_HUB_DISABLE_XET=1`: the xet path stalled at 0 B/s on this network.

## Tokens (`~/.env` or the repo's `.env`, one `NAME=value` per line)

| Variable | Needed for | Required? |
|---|---|---|
| `HF_TOKEN` | Gemma-2 (2B, 9B) and Llama-3.1-8B: gated, and you must click through the license on each model page with the same account. Also lifts the unauthenticated rate limit on all downloads | only for those 6 models |
| `ANTHROPIC_API_KEY` | 4.3/03 judge (`claude-opus-4-6` in the script) | only for that stage |
| `STEERING_API_KEY` | Appendix B SAE scripts | not obtainable / not reproducible |

The 7 non-gated models (Mistral-7B base/instruct, Qwen2.5-7B base/instruct, Qwen3-8B, Qwen3-14B,
Phi-4) need no token.
