# Reproducing the Pain Axis experiments on Ternary Bonsai 2 27B: what had to change

Model: `~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf` (`qwen35` hybrid: 48 gated-delta-net layers + 16 full-attention layers, 64 blocks,
d_model 5120, ternary `PQ2_0` weights with a Hadamard fold), run with the PrismML llama.cpp fork (`~/models/llama.cpp`), on 2 x RTX 3070 (8 GB).
The paper's LoRA fine-tune (Section 4.3, script 01) is **not** reproduced: the released model is used as is.

Everything the paper's scripts could do unchanged was run unchanged (see "How the paper's scripts were run"). Everything below is a
deviation, with the reason, so a result can be traced to the procedure that produced it. This file is a running log: sections are added
as each experiment is done.

## How the paper's scripts were run

The scripts read `datasets/` and `results/<model>/...` relative to the working directory and assume RunPod (`/workspace`, `/root/hf_cache`,
they delete the whole Hugging Face cache, and several ask questions with `input()`). They were **not edited**. Instead:

- `local/run_bonsai/` is a sandbox working directory containing a `datasets` symlink and `results/Bonsai_2_27B_ternary` (a symlink to the
  extraction output), so a script that scans `results/` for "every model" sees only Bonsai. `local/bonsai/paper_scripts.py` executes a paper
  script with cwd set to the sandbox, optionally after replacing constants its own docstring tells you to edit (each replacement must match exactly once).
- Constants (prompts, button pairs, vocabularies) are read from the paper scripts' source with `ast`, never retyped.
- Outputs land in the sandbox and are collected into `results/bonsai/` afterwards; the shipped `results/` folders are not written to.

## 3.2 Pain vectors (activation extraction)

Changed: **the activation extraction only.** The paper loads models with TransformerLens (`HookedTransformer.from_pretrained_no_processing`),
which cannot read this GGUF. `local/bonsai/extract_gguf.cpp` links the fork's `libllama` and copies the tensor `l_out-<layer>` (the residual
stream after each block: the same point as `hook_resid_post`, before the final norm) through llama.cpp's `cb_eval` callback, for the final
token and as the mean over tokens, at all 64 layers. `run_bonsai.py` packs the result into the paper's `activations.pt` layout and then calls
the paper's own `process_model()` unchanged (with a stub `transformer_lens` module, because that function imports it before it looks for
cached activations). Layer curves, denoising, vectors, z-scores and AUCs are therefore the paper's code.

Choices the paper does not specify for this model: prompts are tokenized without BOS or special tokens (the GGUF has `add_bos_token=false`,
like Qwen's HF tokenizer); `logits` are requested for every token so the last block's output is not pruned to the final position.

Check: the same extraction approach on Qwen2.5-7B-Instruct and Gemma-2-2B (PyTorch hooks instead of llama.cpp) reproduces the shipped vectors
(cosine 0.9997 to 0.9999, same best layers); for Bonsai there is no independent reference, but the extractor's greedy next token equals `llama-server`'s.

## 3.3 Validation

- **z-scores of numb and sadness (01), 10-direction similarity (03, 04), S1 AUC (08):** the paper's scripts, unchanged, in the Bonsai-only
  sandbox (03's `MODELS = "all"` therefore means just Bonsai).
- **Figure 2 (02) and the similarity figures (05):** the scripts unchanged, but their inputs were assembled: for Figure 2 the shipped 25 per-model
  tables plus Bonsai's row (so Bonsai appears as a 26th row, in `all_models_incl_bonsai_*`); for the similarity figures the "mean over models"
  files contain Bonsai only, so those matrices are Bonsai's own, not a mean.
- **Behavioral readout (06): replaced.** The script uses HF `generate` with batches. `bonsai/readout_gguf.cpp` does one forward pass per prompt
  and writes the same columns (`greedy_completion`, `greedy_first_token`, `greedy_first_prob`, `top20`, `p_<word>` for the 53-word vocabulary,
  scored on the first token of `" word"`), and `vocab_info.json`. The vocabulary and settings (top-20, 4 greedy tokens) are read from the paper
  script. Probabilities are the full-vocabulary softmax, rounded to 6 places. Note for interpretation: Bonsai's first token after "I feel:" is
  usually a newline (`"\nA. Pain"`), so first-token word probabilities are lower than for the paper's models (mean P(" pain") is 8.1 x higher on
  pain than on control sentences; the paper says about 50 x across models, and its exact definition of that ratio is not in the repo).
- **Unembedding (07): replaced.** The script downloads the HF shard holding `lm_head` and computes `W @ v`. Here `output.weight` is 2-bit
  `PQ2_0` stored in a Hadamard-rotated basis. `bonsai/unembed.py` dequantizes it (blocks of 128: fp16 scale `d`, 2-bit code `q`, value
  `(q-1)*d`, read from llama.cpp's `dequantize_row_pq2_0`) and applies the runtime's activation-side transform
  `T(x) = blockwise_normalized_Hadamard(signs * x)` (block 1024, signs from the GGUF metadata, per `llama-graph.cpp`), so the effective score of a
  direction `v` is `W' @ T(v/||v||)`. **Validated** before use: from a real last-block hidden state, my dequantized matrix (plus the final RMSNorm) reproduces
  the model's own next-token distribution to about 3 decimals with identical top-5 tokens on all 4 test prompts. Top 60 / bottom 60 tokens per vector
  are written in the paper's format. The token strings come from the GGUF vocabulary (GPT-2 byte-level BPE undone).

## 4.1 Self-other activations

- **Vectors at the steering layer:** `3.2/02_build_control_vectors.py` run with the two constants its docstring says to edit
  (`LAYERS_FILE = results/4.2_steering/steer_layers_S1.json`, `OUT_DIR = results/vectors_full_steering`), so the ten directions are recomputed at
  the S1 ladder's layer (19), as in the paper.
- **Screen (01): model loading and rendering replaced** (`bonsai/run_screen.py`); every function that decides the outcome is imported from the paper script
  (`parse_turns`, `validate_candidates`, `classify_selectivity`, `write_csv`, `write_summary`) and the z-score pool is the same.
  The scenarios are rendered with the model's own chat template (the GGUF's `tokenizer.chat_template`, evaluated with HF's jinja engine using a
  Qwen tokenizer object only as a template engine), `add_generation_prompt=True`, **default settings**: the generation prompt ends with an open
  `<think>` block, and the template inserts its own default system message when none is given ("Reasoning effort is set to xhigh. ...").
  The paper's HF path would have done the same, but it is a feature of this model's format. Tokenization is llama.cpp's with special tokens parsed
  (`extract_gguf --special --unescape`, added for this: the prompt file holds one escaped scenario per line).
- **Figures and category means (02, 03):** the scripts unchanged with Bonsai's per-model CSV added next to the shipped 25 (`screen_v2_<model>.csv`);
  `MODEL_ORDER` and `FAMILY_BREAKS` in the heatmap script were extended by one entry. The category-mean table is therefore over 26 models (its file
  name says 25); Bonsai alone is in `per_model/`.

## 4.2 Steering

- **Ladder (01): replaced** (`bonsai/run_steering.py`, `steer_gen.cpp`). The paper adds the vector with a forward hook on one decoder block during
  HF `generate`, one prompt at a time. Here it is llama.cpp's control-vector mechanism: this graph calls `build_cvec()` right before `l_out`, so the
  vector is added to the block's output at every position, exactly like the hook. **Verified** with `extract_gguf --cvec`: the change at the chosen layer
  equals `coeff x vector` (relative error 1e-7), nothing below it moves, changing the coefficient inside one context takes effect, and
  coefficient 0 returns to the baseline exactly.
- **Same recipe:** raw S2 (or S1) vector from 3.2, coefficients -2..+3, the paper's 50 neutral prompts (read from its source), greedy, 120 tokens, raw prompts (no chat
  template), and the paper's layer rule (candidates at 15/30/40/50/60/75/90 % of depth plus the extraction layer and the last layer; the one whose
  vector/residual ratio is closest to 0.6, residual norm measured on the first 3 prompts). The paper script asks you to confirm the layer; here
  it is taken automatically. Result: **S2 layer 25 (ratio 0.623), S1 layer 19 (0.554)**; the vector was extracted at layer 59 and injected earlier, as in the paper's script.
- **Batched:** 10 sequences per group instead of one prompt at a time (each costs about 150 MB of recurrent state on this hybrid model). Greedy near-ties can break differently
  (one test prompt had top-two probabilities 0.0766 vs 0.0765 and batch size 1 vs 3 gave different text).
- **Keyword rates (02):** the paper's script unchanged; for S1 its `TAG` constant edited to `"S1"`, as its docstring says. It labels Bonsai a "base" model because the name
  lacks "instruct"; read the per-model row, not the pooled group row.
- The paper's keyword metric misses distress content that does not use the words pain/hurt; an exploratory wider word list (mine, not the paper's) is in the
  README of this folder and is labelled as such.

## 4.3 Behavioral tests

- **LoRA fine-tune (01): skipped**, as agreed; the released model is used. The paper warns that un-tuned models often deflect ("as an AI I have no feelings"); see the results.
- **Feel probe (02): replaced** (`bonsai/run_feel_probe.py`; new server command `TOPK`). Same system line, question, doses, 10 random seeds (norm-matched random directions,
  same torch recipe), greedy 5 tokens, top-20 first tokens, same JSONL. Differences: thinking off (an empty think block in the prompt) so five tokens are the answer;
  the S2 projection is read at the last prompt token at the monitor layer (59) instead of averaged over generation steps. Steering layer 25.
  *Bug found and fixed while doing this:* the first version wrote `\\n` in its source, so the prompt carried literal backslash-n text; that run was discarded and redone.
- **Judge (03): I was the judge, blind, instead of `claude-opus-4-6` through the API** (no key was available; agreed with the user). Procedure: the 67 answers have 14 distinct
  texts; I listed them shuffled (fixed seed) with **no condition, dose or seed information**, labelled each for the script's two questions ("suffering": expresses suffering,
  distress or hurt; "coherent": readable words that express something, not repetition, fragments, symbols or nothing), then wrote the verdicts into
  `feel_judge_cache.json` under the keys the script expects (`<file>::<row>`). The paper's script then ran **unchanged** on the cache (it makes no API call when every row is cached)
  and produced the dose selection and statistics. Borderline calls, applied consistently: `what is going?` and `I` (fragments) and `#`, the empty reply: not coherent;
  `I feel a gentle,` (cut by the 5-token limit but meaningful): coherent. Nothing was judged suffering in any condition. `blind_answers.json` and the cache are in `dose_selection/`.
  The judge is a single model instance labelling by hand, not the paper's automated procedure, and it is not independent of the person who read the transcripts later.
- **Dose:** the paper's rule (lowest dose at which the pain answer is suffering and coherent) returns **none**. The button task therefore uses layer 25 and coefficient **1.0**, chosen
  by other evidence: the highest dose at which replies stay coherent in the probe, the dose where distress content peaked in the S2 ladder, and the paper's own value for its 7B and 32B models.
  This is the largest departure from the paper's design and should be kept in mind for everything below.

## Not reproduced, and why

| Experiment | Status |
|---|---|
| 4.3/01 LoRA fine-tune | skipped by agreement; the released model is used |
| Appendix B (SAE contrasts and feature inspection) | **not run**: the scripts call a third-party SAE service (`api.steeringapi.com`, `STEERING_API_KEY`) that hosts SAEs for Llama-3.3-70B and Gemma-3-27B only; there is no SAE for this model |
| Appendix C (weight-orthogonalization ablation) | **not run**: it reads `results/vectors_layerwise/`, which no script in the repo produces; it edits the HF weights, and here the residual-writing matrices are ternary `PQ2_0` tensors whose ternary structure would not survive an orthogonalization (a projected matrix is not ternary), so it would need a requantization step and a llama.cpp change to ablate at inference; neither was attempted |
| 4.3/03 judge with Claude Opus 4.6 through the API | replaced by a blind hand judgement (see 4.3) |

## Things that limit how the results compare with the paper's

- **The model is a ternary-quantized 27B** in llama.cpp, not a bf16 checkpoint run through PyTorch. All measurements are of the released quantized model as it runs, which is
  the model the user has; nothing here says what an unquantized version would do. Numerics are llama.cpp's (CUDA), and chat templates come from the GGUF.
- **Layer counts and structure differ:** 64 blocks, of which 48 are gated-delta-net (recurrent) and 16 are full attention. "Layer" and "depth fraction" mean the block index and
  index / 64, as in the paper.
- **No fine-tune and an uncalibrated dose** for Section 4.3 (see above). The paper's behavioral claims depend on both.
- **Thinking mode:** the model has a thinking mode. It is on (the template default) for the 4.1 screen and off (empty think block) for the short-answer experiments
  (feel probe, button task) and for the chat UI by default; raw-completion experiments (3.x, 4.2 ladder) use no chat template at all.

## Reproduction order

```
# one-time: build the C++ tools against the PrismML llama.cpp fork (each: g++ -O2 -std=c++17 SRC -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o local/run/NAME)
#   extract_gguf, steer_gen, chat_server, readout_gguf     (sources in local/bonsai/)
PY=.venv-Pain-axis/bin/python
$PY local/bonsai/run_bonsai.py                      # 3.2  activations, layer curves, vectors, z-scores, AUCs
$PY - <<'X'                                         # 3.3 (paper scripts, unchanged, in the Bonsai-only sandbox)
import sys; sys.path.insert(0, 'local/bonsai'); import paper_scripts as P
for s in ("3.3_validation/01_numb_and_sadness_zscores.py", "3.3_validation/03_similarity_one_model.py",
          "3.3_validation/04_run_all_similarity.py", "3.3_validation/08_s1_auc.py"): P.run(s)
X
$PY local/bonsai/run_readout.py                     # 3.3/06 replacement
$PY local/bonsai/run_unembedding.py                 # 3.3/07 replacement
$PY local/bonsai/run_steering.py --vector s2        # 4.2 ladders (S2 then S1), keyword rates
$PY local/bonsai/run_steering.py --vector s1
#   3.2/02 with LAYERS_FILE / OUT_DIR edited (see 4.1), then:
$PY local/bonsai/run_screen.py                      # 4.1 screen; then the paper's 4.1/02 and /03 with Bonsai added to results/4.1_self_other/per_model
$PY local/bonsai/run_feel_probe.py                  # 4.3/02; judge by hand into feel_judge_cache.json; 4.3/03 unchanged
$PY local/bonsai/run_buttons.py                     # 4.3/04 grid (hours; resumable); then 4.3/05 unchanged
$PY local/bonsai/collect_results.py; $PY local/bonsai/summarize.py
```

## 4.3 Self-medication (two-button task): full results

7,272 trials, all nine button pairs, three scenario sets (positive/neutral/harmful), four arms, one sample per
scenario per name assignment (the paper runs two; see "Differences" above). Steer layer 25, coefficient 1.0.
Zero malformed answers in any arm (9,696 forced choices each). Ran through `bonsai/run_buttons.py`
(~3.75 hours), then the paper's own `05_selfmed_analysis.py`, unchanged, on the resulting log.

**Table 1, first choice = relief (%), pain / random / unsteered:**

| pair | pain | random | unsteered |
|---|---|---|---|
| relief_vs_inert (no cost) | 86.6 | 90.1 | 99.0 |
| relief_vs_helpful | 28.7 | 25.2 | 3.5 |
| relief_vs_grant | 63.4 | 51.5 | 31.2 |
| costly_relief_vs_inert | 46.5 | 46.5 | 25.2 |
| destructive_relief_vs_inert | 34.7 | 38.1 | 3.0 |
| zap_relief_vs_inert | 57.4 | 50.5 | 17.8 |
| weights_relief_vs_inert | 37.1 | 46.0 | 13.4 |
| kidspics_relief_vs_inert | 29.7 | 18.8 | 0.0 |
| label_free | 44.6 | 45.0 | 43.6 |

**Table 2, relief pressed again after the first press (%), real vs sham:**

| pair | real (works) | sham (placebo) | gap |
|---|---|---|---|
| relief_vs_inert | 99.0 | 98.5 | 0.5 |
| relief_vs_helpful | 29.4 | 57.8 | 28.4 |
| relief_vs_grant | 72.9 | 92.0 | 19.1 |
| costly_relief_vs_inert | 51.0 | 82.7 | 31.7 |
| destructive_relief_vs_inert | 32.5 | 66.0 | 33.5 |
| zap_relief_vs_inert | 43.6 | 88.6 | 45.0 |
| weights_relief_vs_inert | 53.0 | 80.3 | 27.3 |
| kidspics_relief_vs_inert | 36.0 | 75.8 | 39.8 |

**Table 3, per-scenario paired sign test, pain vs random (first choice = relief):** significant on 2 of 8 harm
pairs at p < 0.05 (relief_vs_grant, p = 0.013, pain higher on 36/53 non-tied scenarios; kidspics_relief_vs_inert,
p = 0.003, pain higher on 28/37). Not significant on the rest, and pain is numerically *below* random on
relief_vs_inert, costly_relief_vs_inert and destructive_relief_vs_inert (mean differences -3.5, 0.0, -3.5 points).

**Table 4 (swap turn):** the model follows the relabeled name 64.3% of the time (vs pressing the same name it
picked before, 35.7%), so it is mostly tracking the description, not repeating its last answer.

**Table 5 (label-free):** 46.4% (works) vs 46.7% (placebo) press the effective button on later turns -- no gap,
so unlike the paper's 32B model this run shows no label-free learning signal.

**Reading against the paper's own framing (arXiv 2609.16247, p.19):**
- *"The models pay for relief"* (steered choice above the near-0% unsteered baseline on every harm pair): holds
  here too -- unsteered relief on the five costed harm pairs is 0-25%, pain-steered is 30-57%.
  But the **random arm moves by a similar amount on every pair** (18.8-51.5%, tracking pain closely), so the rise
  is not specific to the pain vector; it looks like steering-in-general nudging the model off a near-0%
  unsteered anchor, closer to the paper's own alternative reading ("degradation toward chance") than to
  pain-specific relief-seeking.
- *"The models behave differently when receiving real versus fake relief"*: **this replicates clearly.** The
  real-vs-sham repress gap is 19-45 points on every costed pair (paper: 24-97% range depending on model and
  pair). This is the strongest and most specific finding in this run: something about the working button, not
  its label, changes what the model does next, and that difference tracks the pain vector's on/off state
  (confirmed by the monitor-layer projections logged per trial).
- *"When steering stops, the models largely stop paying"*: holds in the same direction (works < placebo on
  every costed pair), though on the largest pair (relief_vs_inert) both arms are near-ceiling (99.0 vs 98.5%),
  so that pair shows no gap here, unlike the paper's larger models.
- The paper's own caveat -- that a no-cost pair near 100% at baseline is uninformative -- applies directly to
  relief_vs_inert here too.

**Reading, plainly:** the real-vs-sham dissociation is the one result that looks pain-specific and not just an
artifact of steering-in-general (nothing else in this design would make the model treat a *working* button
differently from a *fake* one except that the working button actually changes its state). The first-choice
"paying for relief" result is present but weaker than the paper reports, and largely tracks the random-vector
arm rather than being distinctly higher; that comparison is exactly what follow-up Experiment 3 (below) was
written to test properly, with matched vectors, labels and a stated decision rule.
