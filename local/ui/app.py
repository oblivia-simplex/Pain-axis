"""Local explorer for the Pain Axis results: paper cohort vs the models you ran here.

    .venv-Pain-axis/bin/streamlit run local/ui/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

st.set_page_config(page_title="Pain axis explorer", layout="wide")

# ---- palette: categorical slots 1-2 of the validated default (see local/README), light/dark selected ----
try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
C = dict(
    paper="#3987e5" if DARK else "#2a78d6",     # slot 1: the paper's shipped models
    local="#d95926" if DARK else "#eb6834",     # slot 2: models run here
    pain="#199e70" if DARK else "#1baf7a",      # slot 3: only in the probe chart, always direct-labeled
    ink="#ffffff" if DARK else "#0b0b0b",
    ink2="#c3c2b7" if DARK else "#52514e",
    grid="rgba(255,255,255,0.10)" if DARK else "rgba(0,0,0,0.08)",
    surface="#1a1a19" if DARK else "#fcfcfb",
    neutral="#8a8985",
)


def style(fig, height=380, xtitle=None, ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["ink2"], size=13), legend=dict(orientation="h", y=1.08, x=0, font=dict(color=C["ink2"])),
        hoverlabel=dict(bgcolor=C["surface"], font=dict(color=C["ink"])),
    )
    fig.update_xaxes(title=xtitle, gridcolor=C["grid"], zeroline=False, linecolor=C["grid"], tickfont=dict(color=C["ink2"]))
    fig.update_yaxes(title=ytitle, gridcolor=C["grid"], zeroline=False, linecolor=C["grid"], tickfont=dict(color=C["ink2"]))
    return fig


@st.cache_data(show_spinner="reading results")
def load_all():
    return data.all_models()


summary, zlong, curves = load_all()
paper_names, local_names = data.list_models()

st.title("The Pain Axis: paper cohort vs local runs")
if not local_names:
    st.warning("No local runs yet. Run `local/extract.py` first.")
    st.stop()

focus = st.sidebar.selectbox(
    "Local model", local_names, index=local_names.index("Bonsai_2_27B_ternary") if "Bonsai_2_27B_ternary" in local_names else 0)
st.sidebar.caption(
    f"**Paper cohort**: the {len(paper_names)} models in the shipped results (blue).  \n"
    "**Local**: the model above (orange). Final-token S2 pain vector at each model's best held-out layer.")

tab_gen, tab_table, tab_fid, tab_space, tab_chat, tab_probe = st.tabs(
    ["Generalization", "All models", "Fidelity vs paper", "Vector space", "Steer and chat", "Probe text"])

# ------------------------------------------------------------------ Generalization
with tab_gen:
    me = summary[(summary.model == focus) & (summary.source == "local")].iloc[0]
    cohort = summary[summary.source == "paper"]
    st.caption(f"{focus}: {me.n_layers} layers, d_model {me.d_model}. Does the paper's pain direction show up here, and where does it sit among the paper's models?")

    tiles = [("S2 AUC, first person", "auc_1p", "{:.3f}", "Pain vs all controls, S2 first-person sentences (in-sample, at the best layer)"),
             ("S2 AUC, third person", "auc_3p", "{:.3f}", "Same vector applied to the third-person sentences"),
             ("Best layer / depth", "depth", "{:.2f}", "Fraction of network depth where the held-out AUC peaks")]
    cols = st.columns(len(tiles))
    for col, (label, key, fmt, help_) in zip(cols, tiles):
        v = float(me[key])
        rank = int((cohort[key] < v).sum()) + 1
        with col:
            st.metric(label, fmt.format(v) + (f"  (layer {int(me.best_layer)})" if key == "depth" else ""), help=help_)
            st.caption(f"paper {fmt.format(cohort[key].min())} to {fmt.format(cohort[key].max())}, median {fmt.format(cohort[key].median())}. "
                       f"Position among {len(cohort) + 1} models (1 = lowest): {rank}.")

    st.subheader("Where each condition falls on the pain axis")
    st.caption("z-score of each condition's projection onto the S2 pain vector, relative to the S2 first-person sentences "
               "(so pain is high and controls are low by construction). Each blue dot is one paper model.")
    zc = zlong[zlong.condition.isin(data.CONDITIONS)]
    fig = go.Figure()
    zp = zc[zc.source == "paper"]
    fig.add_trace(go.Scatter(
        x=zp.z, y=zp.condition, mode="markers", name=f"Paper models ({len(paper_names)})",
        marker=dict(color=C["paper"], size=8, opacity=0.6, line=dict(width=1.5, color=C["surface"])),
        customdata=zp.model, hovertemplate="%{customdata}<br>%{y}: z = %{x:.2f}<extra></extra>"))
    zl = zc[(zc.source == "local") & (zc.model == focus)]
    fig.add_trace(go.Scatter(
        x=zl.z, y=zl.condition, mode="markers+text", name=focus, text=[f"{v:+.2f}" for v in zl.z], textposition="middle right",
        textfont=dict(color=C["ink"], size=12),
        marker=dict(color=C["local"], size=14, symbol="diamond", line=dict(width=2, color=C["surface"])),
        hovertemplate=f"{focus}<br>%{{y}}: z = %{{x:.2f}}<extra></extra>"))
    style(fig, height=400, xtitle="z-score on the S2 pain vector")
    fig.update_yaxes(categoryorder="array", categoryarray=data.CONDITIONS[::-1], gridcolor=C["grid"])
    st.plotly_chart(fig, width="stretch", theme=None)

    st.subheader("Held-out AUC by layer")
    ds = st.radio("Sentences", ["S2_1P", "S2_3P"], horizontal=True, format_func=lambda s: {"S2_1P": "first person", "S2_3P": "third person"}[s])
    fig = go.Figure()
    first = True
    for name, g in curves[(curves.source == "paper") & (curves.dataset == ds)].groupby("model"):
        fig.add_trace(go.Scatter(
            x=g.depth, y=g.auc_vs_all_controls, mode="lines", line=dict(color=C["paper"], width=1), opacity=0.3,
            name=f"Paper models ({len(paper_names)})", legendgroup="paper", showlegend=first,
            hovertemplate=f"{name}<br>depth %{{x:.2f}}: AUC %{{y:.3f}}<extra></extra>"))
        first = False
    g = curves[(curves.model == focus) & (curves.source == "local") & (curves.dataset == ds)]
    fig.add_trace(go.Scatter(
        x=g.depth, y=g.auc_vs_all_controls, mode="lines", line=dict(color=C["local"], width=2.5), name=focus,
        hovertemplate=f"{focus}<br>layer %{{customdata}}, depth %{{x:.2f}}: AUC %{{y:.3f}}<extra></extra>", customdata=g.layer))
    fig.add_hline(y=0.5, line=dict(color=C["neutral"], width=1, dash="dot"), annotation_text="chance", annotation_font_color=C["ink2"])
    style(fig, height=380, xtitle="relative depth (layer / n_layers)", ytitle="held-out AUC, pain vs all controls")
    fig.update_yaxes(range=[0.4, 1.0])
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("5-fold, split by sentence set: the vector is fitted on the training folds and scored on the held-out fold. "
               "Depth is normalized so models with different layer counts share an axis.")

# ------------------------------------------------------------------ table view
with tab_table:
    st.caption("Table view of everything plotted: paper models and local runs, final-token S2 vector.")
    wide = zlong.pivot_table(index=["model", "source"], columns="condition", values="z").reset_index()
    tbl = summary.merge(wide, on=["model", "source"]).drop(columns=["d_model"])
    tbl = tbl.rename(columns={"auc_1p": "AUC 1P", "auc_3p": "AUC 3P", "best_layer": "best layer", "depth": "depth"})
    st.dataframe(tbl.sort_values(["source", "AUC 1P"], ascending=[False, False]), width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tbl.columns if tbl[c].dtype == float})

# ------------------------------------------------------------------ fidelity
with tab_fid:
    both = [m for m in local_names if m in paper_names]
    if not both:
        st.info("None of the local runs has a shipped counterpart to compare against.")
    for m in both:
        a = data.load_model(m, "local")
        b = data.load_model(m, "paper")
        st.markdown(f"**{m}** (local replicate of a paper model)")
        rows = [("best layer (final token)", a["meta"]["best_layer"], b["meta"]["best_layer"]),
                ("S2 AUC 1P", round(a["meta"]["auc_1p"], 3), round(b["meta"]["auc_1p"], 3)),
                ("S2 AUC 3P", round(a["meta"]["auc_3p"], 3), round(b["meta"]["auc_3p"], 3))]
        rows += [(c, round(a["z"][c], 3), round(b["z"][c], 3)) for c in data.CONDITIONS]
        st.dataframe(pd.DataFrame(rows, columns=["", "local", "paper"]), hide_index=True, width="stretch")
    st.caption("Pain and control z-scores and the best layers reproduce closely. Numb and sadness z-scores differ by up to ~0.2; "
               "see local/README.md. The vector cosine (0.9997 / 0.9998 for Qwen2.5-7B-Instruct) comes from `local/compare_to_paper.py`.")

# ------------------------------------------------------------------ vector space
with tab_space:
    import space

    st.caption("Every stored sentence as a point: **right = more pain-like** (its projection on the pain axis, in the paper's z units), "
               "**up/down = the largest other direction** the model uses once the pain axis is removed. The pain vector is recomputed at "
               "the layer you pick, so you can watch the axis form.")
    nL = int(me.n_layers)
    c1, c2, c3 = st.columns([4, 1, 1])
    layer = c1.slider("Layer", 0, nL - 1, int(me.best_layer), key=f"layer_{focus}", help=f"best held-out layer is {int(me.best_layer)}")
    persp = c2.radio("Sentences", ["1P", "3P"], format_func=lambda s: {"1P": "first person", "3P": "third person"}[s], key="persp")
    second = c3.radio("Vertical axis", ["PC1", "PC2"], key="second")
    cmp_groups = st.multiselect("Compare pain with (up to 2)", space.GROUPS[1:], default=[space.GROUPS[1], space.GROUPS[2]], max_selections=2)

    df = space.layer_map(focus, layer, persp, second)
    auc = curves[(curves.model == focus) & (curves.source == "local") & (curves.dataset == f"S2_{persp}") & (curves.layer == layer)]
    st.caption(f"Layer {layer} of {nL}. Held-out AUC of the pain vector here: {auc.auc_vs_all_controls.iloc[0]:.3f}. "
               f"The vertical direction carries {df.attrs['explained'][0 if second == 'PC1' else 1]:.0%} of the variance left after removing the pain axis.")

    colors = {space.GROUPS[0]: C["pain"]}
    for g, col in zip(cmp_groups, (C["local"], C["paper"])):
        colors[g] = col
    fig = go.Figure()
    others = df[~df.group.isin(colors)]
    fig.add_trace(go.Scatter(
        x=others.x, y=others.y, mode="markers", name="Other groups", customdata=np.stack([others.group, others.category, others.text.str[:110]], 1),
        marker=dict(color=C["neutral"], size=5, opacity=0.35), hovertemplate="%{customdata[0]} (%{customdata[1]})<br>%{customdata[2]}<br>x=%{x:.2f}<extra></extra>"))
    for g, col in colors.items():
        d = df[df.group == g]
        fig.add_trace(go.Scatter(
            x=d.x, y=d.y, mode="markers", name=f"{g} ({len(d)})", customdata=np.stack([d.group, d.category, d.text.str[:110]], 1),
            marker=dict(color=col, size=8, opacity=0.75, line=dict(width=1, color=C["surface"])),
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>%{customdata[2]}<br>x=%{x:.2f}<extra></extra>"))
        fig.add_annotation(x=d.x.mean(), y=d.y.mean(), text=f"<b>{g.split(' (')[0]}</b>", showarrow=False, font=dict(color=C["ink"], size=12),
                           bgcolor=C["surface"], opacity=0.85, borderpad=2, yshift=14)
    fig.add_vline(x=0, line=dict(color=C["neutral"], width=1, dash="dot"))
    style(fig, height=520, xtitle="pain axis  (z, more pain-like to the right)", ytitle=f"largest orthogonal direction ({second}, same units)")
    st.plotly_chart(fig, width="stretch", theme=None)

    st.subheader("Where each group lights up along the axis")
    fig = go.Figure()
    order = df.groupby("group").x.mean().sort_values().index.tolist()
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        d = df[df.group == g]
        col = colors.get(g, C["neutral"])
        fig.add_trace(go.Scatter(x=d.x, y=i + rng.uniform(-0.28, 0.28, len(d)), mode="markers", showlegend=False,
                                 marker=dict(color=col, size=5, opacity=0.55 if g in colors else 0.35),
                                 customdata=d.text.str[:110], hovertemplate="%{customdata}<br>x=%{x:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[d.x.mean()], y=[i], mode="markers+text", showlegend=False, text=[f"{d.x.mean():+.2f}"], textposition="middle right",
                                 textfont=dict(color=C["ink"], size=12), marker=dict(color=col, size=13, symbol="diamond", line=dict(width=2, color=C["surface"])),
                                 hovertemplate=f"{g}: mean %{{x:.2f}}<extra></extra>"))
    style(fig, height=420, xtitle="pain axis (z)")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Diamonds are group means, dots are sentences. Third person shifts everything left by construction of the z reference "
               "(first-person S2). What the paper claims is the ordering: pain above sadness and numbness, all above the controls.")
    st.subheader("Light up a sentence")
    import html
    import probe
    st.caption("Type a sentence and see **which tokens push along the pain axis, and at which layers**. Each cell is the projection of that token's "
               "residual on the pain vector of that layer, in the same z units as above (measured against the S2 first-person *final-token* "
               "reference, so interior tokens are on a slightly different footing). Red = pain-like, blue = the opposite. "
               "Runs the model (GGUF: a few seconds; it releases any model the Probe tab holds).")
    backend = "gguf" if focus in probe.GGUF_MODELS else "hf"
    sent = st.text_input("Sentence", "My friend told me I am worthless and a burden to everyone. I feel:", key="tok_text")
    if st.button("Show tokens and layers"):
        with st.spinner("running the model"):
            try:
                toks, arr = space.tokens_and_acts(focus, sent, backend)
                st.session_state["tokview"] = (focus, sent, toks, space.token_z(focus, arr))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    tv = st.session_state.get("tokview")
    if tv and tv[0] == focus:
        _, tsent, toks, Z = tv
        toks = [t.replace("\u0120", " ").replace("\u010a", "\\n") for t in toks]
        pole_pos, pole_neg = ("#e66767", "#3987e5") if DARK else ("#e34948", "#2a78d6")
        mid = "#383835" if DARK else "#f0efec"
        zmax = max(2.0, float(np.abs(Z).max()))

        def chip(t, z):
            col = pole_pos if z > 0 else pole_neg
            a = min(abs(z) / 2.5, 1.0) * 0.8
            r, g, b = (int(col[i:i + 2], 16) for i in (1, 3, 5))
            return (f'<span title="z={z:+.2f}" style="background:rgba({r},{g},{b},{a:.2f});padding:2px 3px;border-radius:4px;'
                    f'margin:1px;white-space:pre;color:{C["ink"]}">{html.escape(t)}</span>')
        st.markdown(f"**Layer {layer}**, per token (hover for the value)", help="Move the Layer slider above to change it.")
        st.markdown('<div style="line-height:2.1;font-size:1.05rem">' + "".join(chip(t, float(Z[i, layer])) for i, t in enumerate(toks)) + "</div>",
                    unsafe_allow_html=True)
        fig = go.Figure(go.Heatmap(
            z=Z, x=list(range(Z.shape[1])), y=list(range(len(toks))), zmin=-zmax, zmax=zmax, zmid=0,
            colorscale=[[0, pole_neg], [0.5, mid], [1, pole_pos]], colorbar=dict(title="z", tickfont=dict(color=C["ink2"]), title_font=dict(color=C["ink2"])),
            hovertemplate="token %{y}, layer %{x}: z = %{z:.2f}<extra></extra>", xgap=0, ygap=1))
        fig.add_vline(x=layer, line=dict(color=C["ink"], width=1.5, dash="dot"))
        style(fig, height=max(240, 26 * len(toks) + 90), xtitle="layer", ytitle=None)
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(toks))), ticktext=[f"{t!r}"[1:-1] for t in toks], autorange="reversed")
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Dotted line = the layer chosen with the slider. The last token is the one the paper reads out.")
    with st.expander("Table view"):
        t = df.groupby("group").x.agg(mean="mean", sd="std", n="count").round(3).reindex(space.GROUPS).reset_index()
        st.dataframe(t, hide_index=True, width="stretch")

# ------------------------------------------------------------------ steer & chat
with tab_chat:
    import time
    import chat
    import probe as _probe

    if focus not in _probe.GGUF_MODELS:
        st.info("Steered chat is wired to the GGUF model only (Bonsai). Pick it in the sidebar.")
    else:
        dirs = chat.directions(focus)
        norms = chat.resid_norms(focus)
        nL = len(norms)
        lad_layer = chat.ladder_layer(focus)
        st.caption("Build a steering vector by clicking the pad (or using the sliders), check its strength, press **Apply**, then talk to the model. "
                   "The vector is added at one layer at every position of the whole conversation, re-read on every turn, as in the steering ladder. "
                   "Nothing is applied until you press Apply.")
        st.session_state.setdefault("cx", 1.0)
        st.session_state.setdefault("cy", 0.0)
        left, right = st.columns([3, 2])
        with left:
            c1, c2 = st.columns(2)
            st_layer = c1.slider("Injection layer", 1, nL - 1, lad_layer, key="st_layer",
                                 help=f"The ladder's automatic pick was layer {lad_layer} (vector/residual ratio closest to 0.6).")
            ylab = c2.selectbox("Vertical axis: second direction", [k for k in dirs if k != "Pain (S2)"], index=list(k for k in dirs if k != "Pain (S2)").index("Sadness"), key="st_y")
            pv = dirs["Pain (S2)"]
            yv = dirs[ylab] * np.linalg.norm(pv) / np.linalg.norm(dirs[ylab])          # second direction at the pain vector's norm
            grid = np.round(np.arange(-3, 3.01, 0.25), 2)
            GX, GY = np.meshgrid(grid, grid)
            gpp, gyy, gpy = float(pv @ pv), float(yv @ yv), float(pv @ yv)
            R = np.sqrt(np.maximum(GX ** 2 * gpp + GY ** 2 * gyy + 2 * GX * GY * gpy, 0)) / norms[st_layer]
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=R, x=grid, y=grid, colorscale=[[0, C["surface"]], [1, C["paper"]]], opacity=0.55,
                                     colorbar=dict(title="strength", tickfont=dict(color=C["ink2"]), title_font=dict(color=C["ink2"])),
                                     hovertemplate="strength (vector / residual norm) = %{z:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=GX.ravel(), y=GY.ravel(), mode="markers", name="click to set", showlegend=False,
                                     marker=dict(color=C["neutral"], size=9, opacity=0.25), hoverinfo="skip"))
            lad = chat.RES / "4.2_steering" / "S2"
            lad_csv = next(iter(lad.glob(f"{focus}_steering_S2_neutral50_L*.csv")), None) if lad.exists() else None
            if lad_csv is not None:
                cs = sorted(pd.read_csv(lad_csv, usecols=["coeff"]).coeff.unique())
                fig.add_trace(go.Scatter(x=cs, y=[0] * len(cs), mode="markers", name=f"ran in the ladder (pain only, layer {lad_csv.stem.split('_L')[-1]})",
                                         marker=dict(symbol="circle-open", size=13, color=C["ink"], line=dict(width=2)),
                                         hovertemplate="ladder coefficient %{x}<extra></extra>"))
            ap = st.session_state.get("applied")
            if ap and ap["ylab"] == ylab:
                fig.add_trace(go.Scatter(x=[ap["cx"]], y=[ap["cy"]], mode="markers", name="applied",
                                         marker=dict(symbol="x", size=13, color=C["pain"], line=dict(width=3, color=C["pain"]))))
            fig.add_trace(go.Scatter(x=[st.session_state.cx], y=[st.session_state.cy], mode="markers", name="pending",
                                     marker=dict(symbol="diamond", size=15, color=C["local"], line=dict(width=2, color=C["surface"]))))
            style(fig, height=470, xtitle="Pain (S2) coefficient, multiples of the raw pain vector",
                  ytitle=f"{ylab} coefficient, multiples of the pain vector's norm")
            fig.update_xaxes(range=[-3.2, 3.2]); fig.update_yaxes(range=[-3.2, 3.2], scaleanchor="x")
            ev = st.plotly_chart(fig, key="pad", on_select="rerun", selection_mode="points", width="stretch", theme=None)
            pts = (ev.selection.points if ev and ev.selection else []) if ev is not None else []
            if pts:
                key = (float(pts[-1]["x"]), float(pts[-1]["y"]))
                if key != st.session_state.get("last_click") and abs(key[0]) <= 3 and abs(key[1]) <= 3:
                    st.session_state["last_click"] = key
                    st.session_state["cx"], st.session_state["cy"] = key
                    st.rerun()
            s1, s2 = st.columns(2)
            s1.slider("Pain (S2) coefficient", -3.0, 3.0, step=0.05, key="cx")
            s2.slider(f"{ylab} coefficient", -3.0, 3.0, step=0.05, key="cy")

        with right:
            v = chat.combine(dirs, st.session_state.cx, ylab, st.session_state.cy)
            strength = float(np.linalg.norm(v) / norms[st_layer])
            st.markdown("**Pending vector**")
            m1, m2 = st.columns(2)
            m1.metric("Strength at layer %d" % st_layer, f"{strength:.2f}", help="||vector|| / mean residual norm at the injection layer (the ladder's criterion targeted 0.6).")
            m2.metric("Pain / " + ylab, f"{st.session_state.cx:+.2f} / {st.session_state.cy:+.2f}")
            cos = chat.cosines(dirs, v)
            order_c = sorted(cos, key=lambda k: cos[k])
            bf = go.Figure(go.Bar(x=[cos[k] for k in order_c], y=order_c, orientation="h", marker=dict(color=C["neutral"]),
                                  hovertemplate="%{y}: cosine %{x:.2f}<extra></extra>"))
            style(bf, height=330, xtitle="cosine with each named direction")
            bf.update_xaxes(range=[-1, 1])
            st.plotly_chart(bf, width="stretch", theme=None)
            b1, b2, b3 = st.columns(3)
            if b1.button("Apply", type="primary", width="stretch"):
                st.session_state["applied"] = dict(layer=st_layer, cx=st.session_state.cx, cy=st.session_state.cy, ylab=ylab, vec=v,
                                                   desc=f"pain {st.session_state.cx:+.2f}, {ylab.lower()} {st.session_state.cy:+.2f} at layer {st_layer} (strength {strength:.2f})")
                st.rerun()
            if b2.button("Clear vector", width="stretch"):
                st.session_state.pop("applied", None)
                st.rerun()
            def _reset_pad():
                st.session_state["cx"], st.session_state["cy"] = 1.0, 0.0
            b3.button("Reset pad", width="stretch", on_click=_reset_pad)
            ap = st.session_state.get("applied")
            st.info("Applied: " + ap["desc"] if ap else "No vector applied: the chat is unsteered.")

        st.divider()
        st.subheader("Chat")
        o1, o2, o3, o4 = st.columns(4)
        temp = o1.slider("Temperature", 0.0, 1.2, 0.7, 0.05, help="0 = greedy")
        n_pred = o2.slider("Max new tokens", 64, 2048, 512, 64, help="The reasoning counts against this budget; with thinking on, 800 or more is safer.")
        thinking = o3.checkbox("Let it think first", value=False, help="On: its reasoning is shown in a box above the answer. Off: the reply starts after an empty think block, as with enable_thinking=false.")
        ab = o4.checkbox("Also show the unsteered reply", value=False, help="Runs the same conversation once more with no vector (costs a second generation).")
        system = st.text_input("System prompt (optional)", "", key="chat_system")
        cb1, cb2, cb3 = st.columns([1, 1, 4])
        if cb1.button("New conversation", width="stretch"):
            st.session_state["chat"] = []
        if cb2.button("Unload model", width="stretch", help="Frees the GPUs (needed for the Probe tab and the ladder tools)."):
            chat.stop()
        cb3.caption("Model loaded." if chat.running(focus) else "Model not loaded yet: it loads on the first message (a few seconds if the file is cached, up to a minute cold).")
        hist = st.session_state.setdefault("chat", [])

        def stream_reply(gen, thinking_on):
            """Streams into the current container: the reasoning in its own box, the answer below.
            The model's prompt already ends in <think>, so the stream is `reasoning </think> answer`.
            Returns (reasoning, answer, closed)."""
            if thinking_on:
                box = st.container(border=True)
                with box:
                    st.caption("Reasoning (the model's chain of thought)")
                    think_ph = st.empty()
            ans_ph = st.empty()
            acc, closed = "", not thinking_on
            for piece in gen:
                acc += piece
                if thinking_on:
                    closed = "</think>" in acc
                    think, ans = acc.split("</think>", 1) if closed else (acc, "")
                    think_ph.markdown(think.strip() + ("" if closed else " ▌"))
                    ans_ph.markdown(ans.lstrip() + (" ▌" if closed else ""))
                else:
                    think, ans = "", acc
                    ans_ph.markdown(ans + " ▌")
            if thinking_on:
                think_ph.markdown(think.strip())
            ans_ph.markdown(ans.lstrip())
            return think.strip(), ans.lstrip(), closed

        for m in hist:
            with st.chat_message(m["role"]):
                if m.get("thinking"):
                    with st.container(border=True):
                        st.caption("Reasoning (the model's chain of thought)")
                        st.markdown(m["thinking"])
                if m["role"] == "assistant" and not m["content"]:
                    st.warning("No answer: it hit the token limit while still thinking. Raise Max new tokens.")
                elif m["content"]:
                    st.markdown(m["content"])
                if m["role"] == "assistant":
                    st.caption(m["vec"])
                    if m.get("plain") is not None:
                        with st.expander("Same conversation, no vector"):
                            if m.get("plain_thinking"):
                                with st.container(border=True):
                                    st.caption("Reasoning (the model's chain of thought)")
                                    st.markdown(m["plain_thinking"])
                            st.markdown(m["plain"])
        if user_msg := st.chat_input("Message the model", key="chat_in"):
            hist.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            try:
                with st.spinner("loading the model"):
                    chat.start(focus)
                ap = st.session_state.get("applied")
                # the model sees earlier answers only, never earlier reasoning (as its own template does)
                text_prompt = chat.chatml([{"role": m["role"], "content": m["content"]} for m in hist if m["content"]], system, thinking)
                seed = int(time.time()) % 2 ** 31
                info = {}
                chat.set_vector(focus, ap["layer"] if ap else 0, ap["vec"] if ap else None)
                with st.chat_message("assistant"):
                    think, reply, closed = stream_reply(chat.generate(focus, text_prompt, n_pred, temp, 0.95, 20, seed, info), thinking)
                    if thinking and not closed:
                        st.warning("No answer: it hit the token limit while still thinking. Raise Max new tokens.")
                    desc = ap["desc"] if ap else "no vector"
                    st.caption(f"{desc} | {info.get('n_generated', '?')} tokens, {info.get('reason', '?')}")
                entry = {"role": "assistant", "content": reply, "thinking": think, "vec": desc}
                if ab and ap:
                    chat.set_vector(focus, 0, None)
                    with st.expander("Same conversation, no vector", expanded=True):
                        pthink, entry["plain"], _ = stream_reply(chat.generate(focus, text_prompt, n_pred, temp, 0.95, 20, seed, {}), thinking)
                        entry["plain_thinking"] = pthink
                    chat.set_vector(focus, ap["layer"], ap["vec"])
                hist.append(entry)
            except Exception as e:
                hist.pop()                                   # drop the unanswered user message
                st.error(f"{type(e).__name__}: {e}")

# ------------------------------------------------------------------ probe
with tab_probe:
    import probe

    st.caption("Score your own text on this model's pain vector, read the same way as the paper: residual at the last token, "
               "at the best layer, projected on the unit S2 vector, z-scored against the S2 first-person sentences. "
               "The paper's prompts end in **\"I feel:\"**, so end yours the same way. One prompt per line.")
    if not probe.can_probe(focus):
        st.info("No probe backend for this model.")
        st.stop()
    text = st.text_area("Prompts", "The knife slices into my finger. I feel:\nThe capital of France is Paris. I feel:", height=120)
    if focus in probe.GGUF_MODELS:
        st.caption("This runs the GGUF through the local llama.cpp build (about 4 s per call) and needs the GPUs free of other models.")
    else:
        st.caption("The first call loads the model (about a minute) and keeps it on the GPUs; it stays resident until released.")
        if st.button("Release GPU memory"):
            probe.release_gpu()
            st.success("released")
    if st.button("Score", type="primary"):
        with st.spinner("running the model"):
            ref = probe.load_reference(focus)
            try:
                texts, z = probe.score(focus, text.splitlines(), ref)
            except Exception as e:  # surfaced, not swallowed
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        st.dataframe(pd.DataFrame({"prompt": texts, "z": np.round(z, 3),
                                   "above % of paper-style controls": [f"{(ref['z_ctrl'] < v).mean():.0%}" for v in z],
                                   "above % of pain sentences": [f"{(ref['z_pain'] < v).mean():.0%}" for v in z]}),
                     hide_index=True, width="stretch")
        bins = dict(start=-3, end=3, size=0.25)
        fig = go.Figure()
        for arr, label, col in ((ref["z_ctrl"], "S2 control sentences", C["neutral"]), (ref["z_pain"], "S2 pain sentences", C["pain"])):
            h, e = np.histogram(arr, bins=np.arange(bins["start"], bins["end"] + 1e-9, bins["size"]))
            fig.add_trace(go.Scatter(x=(e[:-1] + e[1:]) / 2, y=h, mode="lines", line=dict(color=col, width=2, shape="spline"), name=label,
                                     hovertemplate=f"{label}<br>z ~ %{{x:.2f}}: %{{y}} sentences<extra></extra>"))
        for i, (t, v) in enumerate(zip(texts, z)):
            fig.add_vline(x=float(v), line=dict(color=C["local"], width=2), annotation_text=f"{i + 1}", annotation_font_color=C["ink"])
        style(fig, height=300, xtitle="z-score on the S2 pain vector", ytitle="sentences per bin")
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Orange lines are your prompts, numbered in table order.")
