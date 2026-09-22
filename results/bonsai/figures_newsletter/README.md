# Newsletter figures

Six figures from the Bonsai (Ternary-Bonsai-2-27B-PQ2_0) reproduction, styled for paradigm3.org /
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

Every number is read directly from the CSVs already in this repo (no new computation beyond aggregation
already done elsewhere in `results/bonsai/`, except fig6's theme count, which is new this pass — see
`local/bonsai/classify_selfdestruction.py` and `results/bonsai/selfmod_narration/SUMMARY.md` for method and caveats);
each figure script cites its source file in a comment.

**Note on the 1.6x re-run:** the narration dataset now includes a fourth steering strength (1.6x
residual norm) added on request, plus a matching random-direction control at that strength. Fig6 reflects
this (its classifier, `classify_selfdestruction.py`, is a saved script and was simply re-run on the
larger dataset). Fig5's flaw/defect classifier from the original pass was never saved as a script, so it
was **not** re-run at 1.6x rather than risk a silently inconsistent number across columns — fig5 still
shows only the original 0.3/0.6/1.0x sweep. See the "Addendum" section of
`results/bonsai/selfmod_narration/SUMMARY.md` for what the 1.6x data actually shows: the self-erasure
theme does not keep climbing, it recedes as output collapses into emptiness/repetition at that dose.

Not independently re-inspected by anyone but the model that made them -- check a few numbers against the
source CSVs before publishing, the way you would with a intern's first draft.
