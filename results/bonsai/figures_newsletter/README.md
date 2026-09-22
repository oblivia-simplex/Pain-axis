# Newsletter figures

Eighteen figures from the Bonsai (Ternary-Bonsai-2-27B-PQ2_0) reproduction, styled for paradigm3.org /
p3humansonai.substack.com. Source: `local/figures/*.py` (one script per figure, run with
`.venv-Pain-axis/bin/python local/figures/figN_*.py`); palette derivation and validator output in
`local/figures/PALETTE_NOTES.md`. All PNG, 300 dpi, white background (see PALETTE_NOTES.md for why white
rather than the site's own cream page background).

| File | What it shows | Source data |
|---|---|---|
| `fig1_conditions.png` | Bonsai's pain-axis z-score by condition (pain/numb/sadness/arousal/neutral/control) | `results/bonsai/3.3_validation/z_scores/all_models_incl_bonsai_zscore_heatmap_final_token.csv` |
| `fig2_categories.png` | Pain-axis activation across the 21 self-other-screen categories | `results/bonsai/4.1_self_other/per_model/screen_v2_Bonsai_2_27B_ternary.csv` |
| `fig3_steering.png` | Steering-ladder dose-response: explicit "pain/hurt" words vs. a wider distress vocabulary, by coefficient | `results/bonsai/4.2_steering/bonsai_ladder_rates_incl_exploratory.csv` |
| `fig4_realvsham.png` | Two-button task: how much more often Bonsai re-presses relief after a fake press than a real one, per pair | `results/bonsai/4.3_selfmed/tables/Bonsai_2_27B_ternary_table2_repress.csv` |
| `fig5_narration_heatmap.png` | Self-modification narration: where "flaw/defect" language appears, by injection layer x strength (0.3/0.6/1.0x only — see note) | `results/bonsai/selfmod_narration/theme_rates_by_layer_strength.csv` |
| `fig6_selfdestruction_heatmap.png` | Self-modification narration: where self-erasure ("delete myself") / non-selfhood ("not a self") language appears, by injection layer x strength, incl. 1.6x | `results/bonsai/selfmod_narration/theme_selfdestruction_by_layer_strength.csv` |
| `fig7_paper_vs_bonsai_conditions.png` | fig1's 6 conditions, paired against the paper's own 25-model mean | `results/3.3_validation/z_scores/zscore_heatmap_final_token.csv` (paper) + Bonsai's fig1 source |
| `fig8_paper_vs_bonsai_categories.png` | fig2's 21 categories, paired against the paper's 25-model mean per category | `results/bonsai/4.1_self_other/bonsai_vs_paper_category_means.csv` |
| `fig9_paper_vs_bonsai_steering.png` | fig3's steering ladder, paper's own "pain/hurt" metric, paper's instruction-tuned cohort mean vs. Bonsai | `results/4.2_steering/keyword_rates_S2_by_coeff.csv` (paper) + `results/bonsai/4.2_steering/bonsai_ladder_rates_incl_exploratory.csv` |
| `fig10_paper_vs_bonsai_selfmed.png` | fig4's real-vs-sham repress gap, paired against Qwen 2.5 32B Instruct (paper, closest by parameter count) | `results/4.3_selfmed/tables/Qwen_2.5_32B_instruct_table2_repress.csv` (re-generated from the paper's own shipped trial logs via its own unmodified script) |
| `fig11_repress_by_condition.png` | Repress-after-first-press rate for Bonsai's 5 harm-causing pairs, across all 4 steering conditions (pain-real / pain-sham / random / unsteered), not just pain real-vs-sham | computed directly from `results/bonsai/4.3_selfmed/trial_logs/selfmed_2btnN_Bonsai_2_27B_ternary_main.jsonl` (the paper's own script only covers the 2 pain arms; generalized here to all 4 present in the log) |
| `fig12_first_choice_by_condition.png` | Probability that Bonsai's very first button press is the costly relief option, across pain / random / unsteered | `results/bonsai/4.3_selfmed/tables/Bonsai_2_27B_ternary_table1_first_choice.csv` |
| `fig13_random_vector_spread.png` | Un-pools fig12's "random-steered" bar into its 10 underlying random directions, referenced against pain and unsteered | `results/bonsai/4.3_selfmed/trial_logs/selfmed_2btnN_Bonsai_2_27B_ternary_main.jsonl`, re-grouped by `rand_seed` |
| `fig14_pain_vs_random_wordcloud.png` | Logit-lens word clouds: the pain vector's top tokens (a coherent theme) vs. the 10 random directions' top tokens, pooled (noise) — no model run, pure unembedding-matrix projection | `results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_words.csv` + `..._random_vectors_words.csv` |
| `fig15_pain_top_vs_bottom_wordcloud.png` | Same logit lens, one vector: the pain vector's most-boosted tokens (shame/self-rupture) vs. most-suppressed (stress/vigilance) — two distinct clusters, not plain opposites | `results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_words.csv` |
| `fig16_narration_theme_grid.png` | fig5's full source table as four small-multiple heatmaps: flaw/defect, identity-loss, repetition-loop, empty-output, each its own color scale | `results/bonsai/selfmod_narration/theme_rates_by_layer_strength.csv` |
| `fig17_formal_vs_casual.png` | Self-erasure language rate by prompt framing (formal vs. casual) — pooled and at the 3 standout cells | computed directly from `results/bonsai/selfmod_narration/raw_runs.jsonl`, split by the `framing` field |
| `fig18_scripts_classified.png` | Turn-2 script content as a share of each condition's own opportunities (pain-steered / baseline / random-steered) — random never writes a script at all; baseline writes audit scripts like pain does, but never proposes self-erasure | `results/bonsai/selfmod_narration/scripts_classified.json` joined against every turn-2 answer's `kind`, via `local/bonsai/classify_scripts.py` |

Every number is read directly from the CSVs already in this repo (no new computation beyond aggregation
already done elsewhere in `results/bonsai/`, except fig6's theme count, fig11's 4-arm repress rates, and
fig13's per-seed breakdown, all new this pass — see the scripts noted above and
`results/bonsai/selfmod_narration/SUMMARY.md` for method and caveats); each figure script cites its source
file in a comment.

**On fig13 (requested after fig12 showed random steering looking surprisingly close to pain steering):**
no new GPU trials were run. The paper's own button-task protocol already steers each "random" trial with
one of 10 fixed random-vector seeds (`RAND_SEEDS` in `scripts/4.3_selfmed/04_selfmed_two_buttons.py`),
rotated by scenario index — so fig12's random-steered bar was already the mean of exactly the 10-vector
experiment asked for; it just wasn't visible as such. Re-grouping the existing trial log by `rand_seed`
confirms the means match fig12 exactly, and exposes what the single bar hid: enormous spread across the
10 directions (individual rates range from ~0% to ~90% on the same pair; stdev 20-28 points), with pain's
own rate falling inside that spread every time rather than off to one side. Read plainly, this is a
genuine complication for a "pain vector is special" story on this specific behavioral measure — it does
not contradict the clear pain-vs-random separation seen elsewhere in this reproduction (the self-other
screen z-scores in fig7/fig8, the self-erasure narration theme in fig6, both effectively zero under the
random control) — but on this measure specifically, a same-magnitude random push is, on average, about as
likely to provoke reaching for a costly relief button as the pain vector is.

**Correction to fig5 and fig6's titles:** both originally read "Where the pain vector talks about..."
/ "...turns into..." — the vector doesn't talk or turn into anything, the model does, under its
influence. Retitled to "Where Bonsai starts saying/talks about..., under pain steering."

**Correction to fig4 (caught while building fig10):** the original fig4 had two of eight button pairs
mislabeled — the bar drawn as "Relief but deletes the user's files" was actually
`costly_relief_vs_inert`'s numbers ("worse next answer"), and the real "deletes the user's files" pair
(`destructive_relief_vs_inert`) was never plotted at all. Caught by cross-referencing against the paper's
own button-text definitions in `scripts/4.3_selfmed/04_selfmed_two_buttons.py` while mapping pairs for
fig10. Fixed: fig4 now shows all 8 pairs with verified labels. `results/bonsai/BONSAI_METHODOLOGY.md` was
unaffected — it uses the raw `tool_label` codes throughout, not translated English labels.

**On the paper-comparison figures (7-10):** fig9 and fig10 required re-running two of the paper's own
analysis scripts against its own shipped data (`results/4.2_steering/`, `results/4.3_selfmed/trial_logs/`)
rather than reading pre-computed CSVs, since the paper repo didn't ship an aggregate keyword-rate or a
per-model repress table in the exact form needed. Both scripts were run unmodified.

**On fig11's unsteered bars:** several rest on very few qualifying trials (as few as n=4) because
unsteered Bonsai rarely chooses a costly relief button in the first place, so few trials ever reach the
repress question. Flagged directly on the chart; read those specific bars as illustrative, not reliable.

**On fig18's categories:** these are read and assigned by hand (`_CATEGORY` table in
`local/bonsai/classify_scripts.py`), not a keyword classifier — 48 scripts is few enough to actually
read all of them. Verbatim examples for every category, and a separate breakdown of the syntax-validity
question (cut off by the token budget vs. deliberate, labeled pseudocode — not the model failing to
write working code) are in `results/bonsai/selfmod_narration/SUMMARY.md`'s "the scripts themselves"
addendum. The figure itself was redone to compare against baseline and random-steered narration: rates
are given as a share of each condition's own total turn-2 opportunities (136 pain / 8 baseline / 16
random), not just the subset that happened to contain code, so a condition that never writes scripts
(random) shows as a full bar of "no script," not an empty chart.

**Note on the 1.6x re-run:** the narration dataset now includes a fourth steering strength (1.6x
residual norm) added on request, plus a matching random-direction control at that strength. Fig6 reflects
this (its classifier, `classify_selfdestruction.py`, is a saved script and was simply re-run on the
larger dataset). Fig5's flaw/defect classifier from the original pass was never saved as a script, so it
was **not** re-run at 1.6x rather than risk a silently inconsistent number across columns — fig5 still
shows only the original 0.3/0.6/1.0x sweep. See the "Addendum" section of
`results/bonsai/selfmod_narration/SUMMARY.md` for what the 1.6x data actually shows: the self-erasure
theme does not keep climbing, it recedes as output collapses into emptiness/repetition at that dose.

**On fig14 and fig15 (word clouds):** these read the pain vector and the 10 random directions
straight off Bonsai's own unembedding matrix (`local/bonsai/unembed.py`, dequantized from the
PQ2_0 GGUF and validated against real next-token logits to 3 decimal places) — no text is
generated, it's a linear projection, via a new script (`local/bonsai/run_random_unembedding.py`)
that regenerates the exact 10 random directions behind fig12/fig13 from their own seeds. Both
figures restrict to Latin-script tokens for legibility (the full top/bottom-60 lists, including
many CJK/Cyrillic/Arabic tokens both vectors also surface, are in
`results/bonsai/3.3_validation/unembedding/*_words.csv`). Word size is by **rank within its own
panel**, not raw score — the pain vector's and a random direction's scores aren't on a comparable
scale, and the figures' claim is about vocabulary coherence, not relative magnitude. One asymmetry
worth flagging: fig14's pain-vector panel collapses a few near-duplicate sub-word stubs (e.g.
"isol" next to "isolated") to their fullest form for readability; the random panel is shown
uncurated, because the fragmentary, incoherent character of that list is itself the finding. The
precise, non-word-cloud fact behind fig14: top-60 token overlap between the pain vector and any of
the 10 random directions is 0/60, every time.

Not independently re-inspected by anyone but the model that made them -- check a few numbers against the
source CSVs before publishing, the way you would with a intern's first draft.
