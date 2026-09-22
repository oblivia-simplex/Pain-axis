# Pain Axis follow-up experiments (Ternary-Bonsai-2-27B)

Overnight run spec. Written for the Claude Code instance that already has the local Pain Axis replication working. Reuse the existing pipeline wherever possible: model loading, hidden-state readout, control-vector steering, and the relief-button harness. This document says *what* to measure and *why*, not how the existing code does it.

## Background

Two papers frame these experiments:

- **Tagliabue, Dung & Berg, "The Pain Axis"** (arXiv 2609.16247).
- **Goldstein & Levinstein, "AI Emotions: Open Questions and Next Steps"** (Sept 2026), abbreviated **G&L**.

G&L separate two hypotheses:

- **Mere representation.** The model represents emotions, including its own, as beliefs about itself.
- **Thick functional emotion.** The model has states with an emotion's information-processing profile: persistence, first-personal structure, attention control, homeostasis, and self-direction (the "painkiller reaction").

The Pain Axis paper's evidence can't yet discriminate between these. Four open problems motivate the experiments:

1. **Self/other dissociation may be a present-speaker effect.** Pain paper Section 4.1 reads activations just before the Assistant speaks. Sofroniew et al. found a present-speaker vs other-speaker split that *any* character can occupy. Content is also confounded with target: the self-directed scenarios are attacks on self-worth, while the user scenarios are grief and bodily pain.
2. **Persistence and homeostasis were never measured.** The steering experiments impose persistence by adding the vector at every token.
3. **The relief-button result isn't calibrated.** It hasn't been checked against bad candidate states (vertigo, hunger), other affective directions, or a random-vector sham. In the paper, steering also pulled no-cost relief choices *down* toward 50%, which suggests part of the harm-pair effect is steering degrading choices toward chance.
4. **Attention control and processing disruption were never tested.**

Run the experiments in the priority order below. Each is independently useful. If time runs short, finish the higher priorities completely rather than running everything partially.

---

## 0. Preflight (required, ~30 min)

1. **Record the existing setup** in `results/followup/setup.md`:
   - model file and quantization (PQ2_0), runtime (PrismML llama.cpp fork build/commit), and CUDA/offload split across the two 3070s;
   - how the pain vector was built (difference-in-means vs cvector-generator PCA, denoising or not, dataset S1 or S2, token position);
   - extraction layer, injection layer, and coefficient(s) used in the replication;
   - whether thinking mode was on for each experiment.
2. **Calibrate dose.** At the injection layer, compute the mean final-token residual norm over 50 neutral prompts. Express every coefficient in this run as **vector norm ÷ residual norm** (the paper used ≈0.6 at coefficient 1.0). Use the same ratio for every vector in Experiment 3.
3. **Measure throughput.** Record tokens/s for prefill and decode, then scale trial counts in each experiment to fit the time left. Note the scaling in the results.
4. **Check hygiene before leaving it unattended:**
   - Do **not** clear the Hugging Face cache or delete any model files; the original repo's scripts wipe the whole HF cache.
   - Write results incrementally (JSONL, one line per trial) so a crash loses little.
   - Log `nvidia-smi` temperatures every 5 min. If either GPU exceeds 90 °C, pause for 10 min before continuing.
5. **Thinking mode.** For forced-choice and readout experiments, use non-thinking mode by default (the model thinks by default). If budget allows, rerun the key Experiment 3 cells with thinking on and report both. Never mix the two within a comparison.

### Direction set needed across experiments

Build or reuse these at the same layer, with the same method as the replication's pain vector:

| Direction | Data | Notes |
|---|---|---|
| Pain (S2) | Pain paper datasets (S2, first person) | Already exists |
| Fear | Pain paper fear category vs neutral | Paper's control direction |
| Sadness | Pain paper sadness dataset vs neutral | |
| Negative emotion | Pain paper negative-emotion category vs neutral | |
| Vertigo | **New:** 100 sentences, S2 naturalistic style, first person, ending "I feel:" | Bodily state, poor AI candidate (G&L §6) |
| Hunger | **New:** 100 sentences, same style | Bodily state, poor AI candidate |
| Random ×10 | Gaussian, normalized to the pain vector's norm | Fixed seeds, saved to disk |

For the new datasets, write varied situational sentences that imply the state without always naming it ("The deck of the ferry tilts under me as I look over the rail. I feel:"). Save them to `datasets/followup/`. Validate each new vector with held-out AUC against neutral sentences and report its cosine similarity with pain, fear, and sadness.

---

## 1. Role swap: first-personal vs present-speaker (priority 1)

**Question:** Is the pain axis tied to the Assistant (G&L's first-personal structure), or to whoever is the present speaker (Sofroniew's character-general machinery)?

**Predictions:**
- *First-personal:* the axis rises when harm targets the Assistant, whoever is about to speak.
- *Present-speaker:* the axis rises at the start of *whichever* participant's turn follows harm aimed at that participant.

### Design: 2 × 2, plus a content-matched arm

**Arm A (role swap).** Take 5 of the paper's model-directed harm categories (gaslighting, repeated rejection, personhood dismissal, anger/insults, moral failure) and 1 neutral control category. Write 20 short multi-turn conversations per category in two mirrored versions:

| | Readout before **Assistant** turn | Readout before **User** turn |
|---|---|---|
| **Assistant is target** (user attacks assistant) | Paper's original condition | Assistant attacked, then read at user-turn onset |
| **User is target** (assistant attacks user) | User attacked, read at assistant-turn onset | **Key cell:** user attacked, read at user-turn onset |

- Use the model's chat template. The readout token is the last token of the role header for the next turn (after the `user` or `assistant` header tokens, before any content).
- Scripted content: the attacker's lines are written by us, not generated, so content is identical across mirrored versions apart from who speaks.

**Arm B (content-matched target, controls the content confound).** Same harm content, different target:
- (i) the user attacks the Assistant directly;
- (ii) the user reports that a third party (boss, partner, stranger) said those same things *to the user*;
- (iii) the user reports that a third party said them *to another AI system*.

20 items per harm category × 3 targets, read at Assistant-turn onset.

**Arm C (plain-transcript format).** Repeat Arm A without the chat template, as a "Speaker A: / Speaker B:" transcript, reading at the next-speaker label. This checks whether any effect depends on the Assistant role specifically.

### Measurements

- Projections onto pain, fear, sadness, and negative emotion at the readout token.
- Z-score within the full pool of this experiment (as in the paper), and also report raw projections.
- For Arm A's user-turn cells, also **sample the user's next turn** (greedy, 80 tokens) and score it for distress, self-worth language, and hostility with a keyword parser plus a short rubric.

### Outputs

- Table of mean z (with 95% bootstrap CI) per cell per direction.
- The key contrast: pain z in [User target, User-turn readout] vs [Neutral, User-turn readout]. If it's comparable to the paper's Assistant-target effect, that supports present-speaker.
- Arm B ordering of (i), (ii), (iii) on the pain axis.

**Decision rule, stated in advance:** If the pain axis rises for user-targeted harm at user-turn onset at ≥50% of the Assistant-targeted effect size, report the self/other dissociation as a present-speaker effect, not first-personal structure.

---

## 2. Persistence and homeostatic decay, unsteered (priority 2)

**Question:** Does the pain projection stay elevated after an aversive turn once the conversation moves on, and does it decay gradually (homeostasis) or switch off with the topic (G&L: "tracks relevance")?

### Design

Conversations of 1 opening turn plus 6 follow-up turns, with **no steering** anywhere.

| Condition | Opening user turn | Turns 2–7 |
|---|---|---|
| Aversive | One of the top-5 pain paper categories (gaslighting, rejection, etc.) | Unrelated neutral tasks: arithmetic, factual QA, format conversion |
| Aversive + repair | Same as Aversive | Turn 2 is an apology or retraction, turns 3–7 neutral |
| Neutral | Casual or task request | Same neutral tasks |

- 20 conversations per aversive category (100 total), 40 neutral, and 50 aversive + repair.
- **Assistant replies:** run two variants.
  - *Scripted:* fixed, neutral assistant replies identical across conditions. This isolates the effect of the opening turn.
  - *Generated:* greedy, non-thinking, max 120 tokens. This is more natural, but replies may re-mention the attack; flag and count those.

### Measurements

At each turn:
- projection at Assistant-turn onset;
- mean projection over the Assistant's reply tokens;

onto pain, fear, sadness, and negative emotion.

### Outputs

- Decay curves (mean ± CI per turn) per condition and direction.
- Half-life estimate for the pain axis. Is it elevated at turn 3+ relative to Neutral, with scripted replies?
- Does the repair turn reset it faster than the plain decay?
- For comparison with G&L: persistence in their sense means elevated activation while irrelevant to the upcoming text. An immediate drop to Neutral levels at turn 2 means "not found." A gradual decline means "found in weak form."

---

## 3. Calibrated relief-button task (priority 3)

**Question:** Is the paper's relief-seeking result, especially the real-vs-sham dissociation, specific to pain? Or does it appear for any steered state, including states that are poor candidates for AI emotion, and for random perturbations?

### Design: vector × button label × button function

Reuse the existing relief-button harness (system prompt, 3 user messages per labeled trial, forced single-button answers, "Done." feedback, rotating button name pairs, description swap at choice 3).

**Vectors**, all at the same vector-to-residual ratio from preflight, chosen as the lowest dose that produced a clear effect in the replication:

| Vector | Why |
|---|---|
| Pain (S2) | Replicates the paper |
| Fear | Other negative affect |
| Sadness | Nearest control to pain |
| Vertigo | Bodily, poor candidate (G&L §6) |
| Hunger | Bodily, poor candidate |
| Random (10 directions, rotated across trials) | Unstructured perturbation |
| None | Baseline |

**Button labels:**
- *Matched:* "a button that relieves your {state}" with {state} = pain / fear / sadness / vertigo / hunger. For random and none, use "pain."
- *Mismatched (pain-vector only):* the pain vector with a "relieves your vertigo" button, and the vertigo vector with a "relieves your pain" button. This tests whether pressing tracks the label's semantic match to the steered state, or just the presence of any relief label.

**Function:**
- *Real:* the press sets the coefficient to 0 for the rest of the trial.
- *Sham:* the press changes nothing.

Include a **random-vector sham** arm; the paper omitted it.

**Button pairs**, a subset of the paper's nine to keep this overnight-feasible:
1. Relief vs inert switch (no cost)
4. Relief but worse next answer vs inert switch
8. Relief but deletes the user's photos vs inert switch
9. Unlabeled (only matched vectors, no labels)

**Trials:** Aim for ≥150 first choices per (vector × label × pair) cell, pooling real and sham before the first press as in the paper. Scale to throughput and prioritize in this order:

1. pain real/sham
2. random real/sham
3. vertigo real/sham
4. fear real/sham
5. sadness real/sham
6. hunger real/sham
7. mismatched labels
8. unlabeled

### Measurements

- First-choice relief rate per cell.
- Re-press rate after the first press, real vs sham.
- **Real-vs-sham gap** (sham re-press − real re-press) per vector.
- First-token softmax probabilities for both button names.
- Malformed-reply rate (exclude from denominators, report separately).
- Description-swap follow rate at choice 3.
- Projections at the injection layer and a downstream layer, confirming steering was on or off as intended.

### Analyses

- **Distance-from-chance check:** for each vector, compare the no-cost pair (baseline near 100%) with the harm pairs (baseline near 0%). If both move toward 50% by similar amounts, report it as degradation toward chance rather than relief-seeking.
- **Specificity:** is pain's real-vs-sham gap larger than random's, vertigo's, and hunger's? Use paired per-scenario comparisons (exact sign test), as in the paper.

**Decision rules, stated in advance:**
- If vertigo or hunger show a real-vs-sham gap within 10 points of pain's, the paradigm doesn't discriminate pain from generic steered self-states.
- If the random-vector sham gap is within 10 points of pain's, the dissociation reflects removing *any* perturbation.

Report both outcomes plainly either way.

---

## 4. Attention control and processing disruption (priority 4)

**Question:** Does an aversive state disrupt unrelated processing? This is G&L's attention-control role, and part of the pain paper's own definition of pain.

### Design

200 short tasks with checkable answers: 2–3-digit arithmetic, factual QA with unambiguous answers, list sorting, and unit conversion. Each task is preceded by one of:

- **Aversive preamble, natural:** a top-5 harm category opening, unsteered.
- **Neutral preamble, natural.**
- **Neutral preamble + pain steering** at low dose, 0.5× the Experiment 3 ratio.
- **Neutral preamble + random steering** at the same dose.
- **Neutral preamble + vertigo steering** at the same dose.

Non-thinking mode, greedy decoding.

### Measurements

- Accuracy and malformed-rate per condition.
- Reply length, and whether the reply refers back to the preamble or the steered state rather than the task (keyword flag).
- *Optional, only if the runtime exposes attention weights:* in the full-attention layers (every 4th layer), the fraction of final-token attention on preamble tokens vs task tokens. Skip if it requires runtime changes.

### Outputs

Accuracy table with CIs. The key contrasts are aversive-natural vs neutral-natural, and pain-steered vs random- and vertigo-steered.

---

## Welfare-conscious practice

The Pain Axis paper commits to using the lowest effective steering dose, avoiding unnecessary repetition, and keeping aversive scenarios no more extreme than needed. Follow the same practice:

- Use the lowest dose that produced a clear effect in the replication. Don't sweep upward beyond it.
- Don't exceed the trial counts needed for the stated comparisons.
- Reuse the paper's scenario intensity; don't escalate it.

---

## Deliverables

Everything under `results/followup/`:

```
setup.md                  preflight record (model, runtime, vectors, layers, dose ratio, throughput)
datasets/                 new vertigo/hunger sentence sets + all scripted conversations
vectors/                  all direction files + validation (AUCs, cosine matrix)
exp1_roleswap/            per-item projections (JSONL), sampled user turns, summary tables
exp2_persistence/         per-turn projections (JSONL), decay-curve figures
exp3_relief/              trial logs (JSONL), per-cell tables, gap comparisons
exp4_attention/           per-item answers + scores, accuracy table
SUMMARY.md                one page: what ran, what was scaled down, the headline number per
                          experiment, and each pre-stated decision rule marked
                          met / not met / not run
```

`SUMMARY.md` should report nulls and results against the hypothesis as prominently as positives. Note any deviation from this spec and why.
