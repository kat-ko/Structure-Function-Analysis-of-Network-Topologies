#!/usr/bin/env python3
"""Generate notebooks/10_even_novel_analysis.ipynb (novel + even angles, post-nb09 default)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB07 = ROOT / "notebooks" / "07_overnight_analysis.ipynb"
OUT = ROOT / "notebooks" / "10_even_novel_analysis.ipynb"

src = json.loads(NB07.read_text())
cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

md("""# toy_task — Even-angle novel analysis (nb10)

Focused analysis of the overnight sweep **restricted to `stimulus_regime=novel`**
with **patch-faithful even object angles** (post-nb09 default). Compare against
nb08 (legacy random-angle runs).

Data: `data/runs_overnight/` (runs with `_angeven_` in `run_id`).
Figures: `figures/nb10/`.
""")

md("## 1. Setup — paths, factor levels, colour/style conventions")

code("""import sys, json, glob, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm.auto import tqdm

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import toy_task
from toy_task import (
    CONSTANTS, RunConfig, SIMILARITY_GRID, GAMMA_GRID,
    PAPER_SIM_IDX, PAPER_SIM_LABELS, SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR,
    ANGLE_EVEN,
)
from toy_task import analysis as ana
from toy_task.storage import load_result

OUT_ROOT = PROJECT_ROOT / "data" / "runs_overnight"
FIG_DIR = PROJECT_ROOT / "figures" / "nb10"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE = PROJECT_ROOT / "data" / "novel_even_summary.pkl"
EPOCHS = 8000
REGIME = "novel"   # nb10: novel stimuli + even angles
ANGLE_MODE = ANGLE_EVEN

ARCH_CONFIGS = [
    ("dense", "freeze"),
    ("modular_shared", "freeze"),
    ("modular_feature_routed", "freeze"),
    ("modular_task_routed", "freeze"),
    ("modular_task_routed", "readout_coord"),
]
def arch_label(arch, off):
    return f"{arch}/{off}" if arch == "modular_task_routed" else arch
ARCH_ORDER = [arch_label(a, o) for a, o in ARCH_CONFIGS]
ARCH_DISPLAY = {
    "dense": "Single",
    "modular_shared": "Mod-shared",
    "modular_feature_routed": "Mod-feature (ctrl)",
    "modular_task_routed/freeze": "Mod-task(freeze)",
    "modular_task_routed/readout_coord": "Mod-task(coord)",
}

GAMMAS = sorted(GAMMA_GRID)
HIDDEN_SIZES = [12, 24, 50, 100]
SCOPES = ["all", "no_readout"]
REF_H, REF_SCOPE = 50, "all"
COND = PAPER_SIM_LABELS
PAPER_IDX = PAPER_SIM_IDX
SAME_IDX, NEAR_IDX, FAR_IDX = SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR
ACC_THRESH_DEG = 22.5

G_COLOR = {g: c for g, c in zip(GAMMAS, plt.cm.viridis(np.linspace(0, 1, len(GAMMAS))))}
A_COLOR = {lab: c for lab, c in zip(ARCH_ORDER, plt.cm.tab10(np.linspace(0, 1, 10)))}
COND_COLOR = {"Same": "#ccd9ec", "Near": "#7fa6d9", "Far": "#274b8f"}

def mean_ci(series):
    s = pd.Series(series).dropna().values
    if len(s) == 0:
        return np.nan, np.nan
    m = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0
    return m, se

def g_handles():
    return [Line2D([0], [0], color=G_COLOR[g], lw=2, label=f"γ={g:g}") for g in GAMMAS]
def a_handles():
    return [Line2D([0], [0], color=A_COLOR[l], lw=2, label=ARCH_DISPLAY[l]) for l in ARCH_ORDER]

print("toy_task", toy_task.__version__)
print("regime:", REGIME, "| angle_mode:", ANGLE_MODE, "| cache:", OUT_ROOT.exists())
print(f"γ={GAMMAS} | H={HIDDEN_SIZES} | ref slice: H={REF_H}, scope={REF_SCOPE}")
""")

md("""## 2. Load runs → cached summary (novel + even only)

Reuses the nb07 summary builder, restricted to runs whose ``run_id`` contains
``_angeven_``, then filters to ``stimulus_regime=novel``.
""")

loader_src = "".join(src["cells"][4]["source"])
loader_src = loader_src.replace(
    'CACHE = PROJECT_ROOT / "data" / "overnight_summary.pkl"',
    'CACHE = PROJECT_ROOT / "data" / "novel_even_summary.pkl"',
)
loader_src = loader_src.replace(
    'run_ids = [os.path.basename(os.path.dirname(p))\n'
    '               for p in glob.glob(str(OUT_ROOT / "*" / "config.json"))]',
    'run_ids = [os.path.basename(os.path.dirname(p))\n'
    '               for p in glob.glob(str(OUT_ROOT / "*_angeven_*" / "config.json"))]',
)
loader_src = loader_src.replace(
    "df = build_summary()",
    'df = build_summary()\ndf = df[df.regime == REGIME].copy()\nprint(f"filtered to {ANGLE_MODE} / {REGIME}: {len(df)} runs")',
)
code(loader_src)

md("""## 3. Training sanity — did 8000 epochs/phase converge?

Final angular error by γ×architecture, plus a representative dense/Near loss timecourse.
""")

code("""def load_curve(arch, off, g, si, H=REF_H, scope=REF_SCOPE, seed=0):
    rid = RunConfig(arch=arch, hidden_size=H, gamma=g, similarity=SIMILARITY_GRID[si],
                    seed=seed, off_module_policy=off, epochs_per_phase=EPOCHS,
                    similarity_index=si, init_scope=scope, stimulus_regime=REGIME).run_id()
    ext = load_result(OUT_ROOT, rid)["extractions"]
    o = np.argsort(ext["global_epoch"])
    return ext["global_epoch"][o], ext["clean_loss_T0"][o], ext["clean_loss_Ts"][o]

core = df[(df.H == REF_H) & (df.scope == REF_SCOPE)]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

for ax, col, ttl in zip(axes[:2], ["err_A1_end", "err_B_end"],
                        ["Post-A1 (task A)", "Post-B (task B)"]):
    for lab in ARCH_ORDER:
        agg = core[core.arch_lab == lab].groupby("gamma")[col].mean().reindex(GAMMAS)
        ax.plot(range(len(GAMMAS)), agg.values, marker="o", color=A_COLOR[lab],
                label=ARCH_DISPLAY[lab])
    ax.set_xticks(range(len(GAMMAS))); ax.set_xticklabels([f"{g:g}" for g in GAMMAS])
    ax.set_xlabel("init scale γ  (rich → lazy)"); ax.set_title(ttl)
    ax.axhline(2.0, ls=":", c="grey", lw=0.8)
axes[0].set_ylabel("mean angular error (°)")
axes[0].legend(fontsize=7, ncol=1)

axc = axes[2]
for g, lsty in [(min(GAMMAS), "-"), (max(GAMMAS), "--")]:
    try:
        ge, l0, ls = load_curve("dense", "freeze", g, NEAR_IDX)
        axc.plot(ge, l0, lsty, color="tab:green", lw=1.2, label=f"task A (γ={g:g})")
        axc.plot(ge, ls, lsty, color="tab:purple", lw=1.2, label=f"task B (γ={g:g})")
    except Exception as e:
        axc.text(0.5, 0.5, f"curve missing\\n{e}", ha="center", transform=axc.transAxes)
axc.axvline(EPOCHS, ls=":", c="grey", lw=0.8); axc.axvline(2 * EPOCHS, ls=":", c="grey", lw=0.8)
axc.set_yscale("log"); axc.set_xlabel("global epoch"); axc.set_ylabel("clean MSE")
axc.set_title(f"Dense, Near — loss timecourse ({REGIME})"); axc.legend(fontsize=6)

fig.suptitle(f"Fig 0 — Training sanity ({REGIME})", y=1.03)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig0_training_sanity.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 4. Interference & switch cost vs task similarity

*(RQ1: rich/lazy × architecture, novel stimuli)*  
Reference slice H=50, scope=all.
""")

code("""def agg_metric(sub, metric):
    grp = sub.groupby("similarity")[metric]
    m = grp.mean()
    return m.index.values, m.values, grp.sem().reindex(m.index).fillna(0).values

core2 = df[(df.H == REF_H) & (df.scope == REF_SCOPE)]
metrics = [("interf_deg", "interference (°)\\nerr(A2-start) − err(A1-end)"),
           ("transfer_deg", "switch cost (°)\\nerr(B-start) − err(A1-end)")]

fig, axes = plt.subplots(2, len(ARCH_ORDER), figsize=(3.0 * len(ARCH_ORDER), 6.4),
                         sharex=True, sharey="row", squeeze=False)
for col, lab in enumerate(ARCH_ORDER):
    for row, (metric, ylab) in enumerate(metrics):
        ax = axes[row, col]
        for g in GAMMAS:
            sub = core2[(core2.arch_lab == lab) & (core2.gamma == g)]
            if sub.empty:
                continue
            x, m, se = agg_metric(sub, metric)
            ax.errorbar(x, m, yerr=se, color=G_COLOR[g], marker="o", ms=3, lw=1.1, capsize=1.5)
        ax.axhline(0, c="grey", lw=0.6)
        if row == 0:
            ax.set_title(ARCH_DISPLAY[lab], fontsize=9)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=8)
        if row == len(metrics) - 1:
            ax.set_xlabel("similarity s")
fig.legend(handles=g_handles(), title="init scale γ", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 1 — Interference & switch cost vs similarity ({REGIME})", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_interference_transfer.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("## 5. Interference heatmaps — γ × similarity, per architecture")

code("""sims = sorted(core2.similarity.unique())
vmin = float(min(0.0, core2["interf_deg"].min()))
vmax = float(core2["interf_deg"].quantile(0.98))

fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(2.5 * len(ARCH_ORDER), 2.8), squeeze=False)
im = None
for c, lab in enumerate(ARCH_ORDER):
    ax = axes[0, c]
    sub = core2[core2.arch_lab == lab]
    M = (sub.groupby(["gamma", "similarity"])["interf_deg"].mean()
         .unstack("similarity").reindex(index=GAMMAS, columns=sims))
    im = ax.imshow(M.values, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(ARCH_DISPLAY[lab], fontsize=8)
    ax.set_yticks(range(len(GAMMAS))); ax.set_yticklabels([f"{g:g}" for g in GAMMAS], fontsize=6)
    ax.set_xticks(range(len(sims))); ax.set_xticklabels([f"{s:.2f}" for s in sims], fontsize=5, rotation=90)
    ax.set_ylabel("γ (rich→lazy)" if c == 0 else "")
    ax.set_xlabel("similarity s", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="interference (°)")
fig.suptitle(f"Fig 2 — Interference landscape γ×s ({REGIME}, H=50)", y=1.02)
fig.savefig(FIG_DIR / "fig2_interference_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 6. Forward transfer to unseen stimuli

Holton-faithful readout: error on task B's *novel* objects using the A-trained network
*before* any B updates (`err_B_start`). Lower = the abstract rule generalised.
""")

code("""fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(3.0 * len(ARCH_ORDER), 3.4),
                         sharey=True, squeeze=False)
for col, lab in enumerate(ARCH_ORDER):
    ax = axes[0, col]
    sub = core2[core2.arch_lab == lab]
    for g in GAMMAS:
        x, m, se = agg_metric(sub[sub.gamma == g], "err_B_start")
        ax.errorbar(x, m, yerr=se, color=G_COLOR[g], marker="o", ms=3, lw=1.1, capsize=1.5)
    ax.axhline(90, ls=":", c="red", lw=0.7)
    ax.set_title(ARCH_DISPLAY[lab], fontsize=9)
    ax.set_xlabel("similarity s")
    if col == 0:
        ax.set_ylabel("forward transfer\\nerr(B-start) (°)", fontsize=8)
fig.legend(handles=g_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 3 — Forward transfer on novel stimuli ({REGIME})", y=1.03)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_forward_transfer.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 7. Directional rule-shift metric

`pull_fraction` at A2-start: does forgetting drift A toward B's rule?
Mid-range similarities only (Same undefined, Far degenerate).
""")

code("""rs_idx = [1, 2, 3, 4, 5]
fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(3.0 * len(ARCH_ORDER), 3.4),
                         sharey=True, squeeze=False)
for col, lab in enumerate(ARCH_ORDER):
    ax = axes[0, col]
    for g in GAMMAS:
        sub = core2[(core2.arch_lab == lab) & (core2.gamma == g) & (core2.sim_idx.isin(rs_idx))]
        if sub.empty:
            continue
        x, m, se = agg_metric(sub, "rs_pull")
        ax.errorbar(x, m, yerr=se, color=G_COLOR[g], marker="o", ms=3, lw=1.1, capsize=1.5)
    ax.axhline(1.0, ls=":", c="grey", lw=0.7); ax.axhline(0.0, c="grey", lw=0.6)
    ax.set_title(ARCH_DISPLAY[lab], fontsize=9); ax.set_xlabel("similarity s")
    if col == 0:
        ax.set_ylabel("rule pull fraction\\n0=keep A · 1=on B's rule", fontsize=8)
fig.legend(handles=g_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 4 — Directional rule-shift ({REGIME})", y=1.03)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_rule_shift.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 8. Representational geometry

**Fig 5** — geometry vs γ (participation ratio, principal angle A1↔B, hidden drift).
**Fig 5b** — principal angle vs similarity, columns = γ.
""")

code("""panels = [("pr_A1", "Participation ratio (A1-end)"),
          ("pang_A1B", "Principal angle A1↔B (°)"),
          ("drift_A_overB", "Hidden drift  A1-end→A2-start")]
geo = core2

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), squeeze=False)
for c, (col, ttl) in enumerate(panels):
    ax = axes[0, c]
    for lab in ARCH_ORDER:
        sub = geo[geo.arch_lab == lab].groupby("gamma")[col]
        m = sub.mean().reindex(GAMMAS); se = sub.sem().reindex(GAMMAS).fillna(0)
        ax.errorbar(range(len(GAMMAS)), m.values, yerr=se.values, color=A_COLOR[lab],
                    marker="o", ms=4, lw=1.3, capsize=2, label=ARCH_DISPLAY[lab])
    ax.set_xticks(range(len(GAMMAS))); ax.set_xticklabels([f"{g:g}" for g in GAMMAS])
    ax.set_xlabel("γ (rich → lazy)"); ax.set_title(ttl, fontsize=9)
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Fig 5 — Geometry vs learning regime ({REGIME})", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_geometry_vs_gamma.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""gammas_lazy_first = sorted(GAMMAS, reverse=True)
fig, axes = plt.subplots(1, len(gammas_lazy_first),
                         figsize=(2.6 * len(gammas_lazy_first), 3.2), sharey=True, squeeze=False)
for c, g in enumerate(gammas_lazy_first):
    ax = axes[0, c]
    for lab in ARCH_ORDER:
        sub = geo[(geo.arch_lab == lab) & (geo.gamma == g)]
        if sub.empty:
            continue
        x, m, se = agg_metric(sub, "pang_A1B")
        ax.errorbar(x, m, yerr=se, color=A_COLOR[lab], marker="o", ms=3, lw=1.2, capsize=1.5)
    tag = " (lazy)" if g == max(GAMMAS) else (" (rich)" if g == min(GAMMAS) else "")
    ax.set_title(f"γ={g:g}{tag}", fontsize=9); ax.set_xlabel("similarity s")
    if c == 0:
        ax.set_ylabel("Principal angle A1↔B (°)")
fig.legend(handles=a_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 5b — Subspace reorientation vs similarity ({REGIME})", y=1.03)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig5b_principal_angle.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("## 9. Capacity sweep — hidden size H")

code("""cap = df[(df.scope == "all") & (df.sim_idx.isin(PAPER_IDX))]
cap_metrics = [("interf_deg", "interference (°)"), ("npc99_A1", "#PCs 99% (A1-end)")]

fig, axes = plt.subplots(2, len(ARCH_ORDER), figsize=(3.0 * len(ARCH_ORDER), 6.2),
                         sharex=True, sharey="row", squeeze=False)
for col, lab in enumerate(ARCH_ORDER):
    for row, (metric, ylab) in enumerate(cap_metrics):
        ax = axes[row, col]
        for g in GAMMAS:
            sub = cap[(cap.arch_lab == lab) & (cap.gamma == g)]
            if sub.empty:
                continue
            grp = sub.groupby("H")[metric]
            m = grp.mean().reindex(HIDDEN_SIZES); se = grp.sem().reindex(HIDDEN_SIZES).fillna(0)
            ax.errorbar(range(len(HIDDEN_SIZES)), m.values, yerr=se.values, color=G_COLOR[g],
                        marker="o", ms=3, lw=1.0, capsize=1.5)
        ax.set_xticks(range(len(HIDDEN_SIZES))); ax.set_xticklabels(HIDDEN_SIZES)
        if row == 0:
            ax.set_title(ARCH_DISPLAY[lab], fontsize=9)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=8)
        if row == len(cap_metrics) - 1:
            ax.set_xlabel("hidden size H")
fig.legend(handles=g_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 6 — Capacity sweep ({REGIME})", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_capacity.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("## 10. Readout-scaling ablation — `scope=all` vs `no_readout`")

code("""SCOPE_STYLE = {"all": "-", "no_readout": "--"}
SCOPE_MARK = {"all": "o", "no_readout": "x"}
abl = df[(df.H == REF_H) & (df.sim_idx.isin(PAPER_IDX))]

fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(3.0 * len(ARCH_ORDER), 3.4),
                         sharey=True, squeeze=False)
for col, lab in enumerate(ARCH_ORDER):
    ax = axes[0, col]
    for g in GAMMAS:
        for scope in SCOPES:
            sub = abl[(abl.arch_lab == lab) & (abl.gamma == g) & (abl.scope == scope)]
            if sub.empty:
                continue
            x, m, se = agg_metric(sub, "interf_deg")
            ax.errorbar(x, m, yerr=se, color=G_COLOR[g], ls=SCOPE_STYLE[scope],
                        marker=SCOPE_MARK[scope], ms=4, lw=1.1, capsize=1.5)
    ax.set_title(ARCH_DISPLAY[lab], fontsize=9); ax.set_xlabel("similarity s")
    if col == 0:
        ax.set_ylabel("interference (°)", fontsize=8)
handles = g_handles() + [Line2D([0], [0], color="k", ls=SCOPE_STYLE[s],
                                 marker=SCOPE_MARK[s], label=s) for s in SCOPES]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 7 — Readout-scaling ablation ({REGIME}, H=50)", y=1.03)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig7_scope_ablation.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("## 11. 3D PCA of object representations (drift view)")

code("""PCA_ARCH, PCA_OFF, PCA_SEED = "dense", "freeze", 0
PHASE_COLOR = {"A1": "tab:orange", "B": "tab:blue", "A2": "tab:green"}
rich_g, lazy_g = min(GAMMAS), max(GAMMAS)

def phase_end_hidden(arch, off, g, si, phase, H=REF_H, scope=REF_SCOPE, seed=PCA_SEED):
    rid = RunConfig(arch=arch, hidden_size=H, gamma=g, similarity=SIMILARITY_GRID[si],
                    seed=seed, off_module_policy=off, epochs_per_phase=EPOCHS,
                    similarity_index=si, init_scope=scope, stimulus_regime=REGIME).run_id()
    ext = load_result(OUT_ROOT, rid)["extractions"]
    ph = ext["phase"].astype(str); ep = ext["epoch"].astype(int)
    w = np.where(ph == phase)[0]; i = int(w[np.argmax(ep[w])])
    return ext["hidden"][i]

def loop(ax, pts, color):
    p = np.vstack([pts, pts[:1]])
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color=color, ms=2.5, lw=0.9)

row_specs = [(rich_g,), (lazy_g,)]
cond = [SAME_IDX, NEAR_IDX, FAR_IDX]
fig = plt.figure(figsize=(2.8 * 3, 2.4 * len(row_specs)))
for r, (g,) in enumerate(row_specs):
    for c, si in enumerate(cond):
        ax = fig.add_subplot(len(row_specs), 3, r * 3 + c + 1, projection="3d")
        try:
            hA1 = phase_end_hidden(PCA_ARCH, PCA_OFF, g, si, "A1")
            hB = phase_end_hidden(PCA_ARCH, PCA_OFF, g, si, "B")
            hA2 = phase_end_hidden(PCA_ARCH, PCA_OFF, g, si, "A2")
            _, qA1, qB, qA2 = ana.shared_pca_three_phase(hA1, hB, hA2, n_components=3)
            for q, ph in [(qA1, "A1"), (qB, "B"), (qA2, "A2")]:
                loop(ax, q, PHASE_COLOR[ph])
        except Exception:
            ax.text2D(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        if r == 0:
            ax.set_title(COND[si], fontsize=9)
        if c == 0:
            reg_tag = "rich" if g == rich_g else "lazy"
            ax.text2D(-0.2, 0.5, f"{reg_tag} (γ={g:g})", transform=ax.transAxes,
                      rotation=90, va="center", fontsize=7)
handles = [Line2D([0], [0], color=PHASE_COLOR[p], marker="o", lw=1.2, label=f"Post-{p}")
           for p in ["A1", "B", "A2"]]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
fig.suptitle(f"Fig 8 — 3D PCA ({ARCH_DISPLAY[arch_label(PCA_ARCH, PCA_OFF)]}, {REGIME})", y=1.0)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig8_pca_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("## 12. Quantitative digest")

code("""ref = core2
pd.set_option("display.width", 120)

print("=" * 70)
print(f"A) Rich vs Lazy interference by architecture (°)  [{REGIME}]")
tblA = (ref[ref.gamma.isin([min(GAMMAS), max(GAMMAS)])]
        .groupby(["arch_disp", "gamma"])["interf_deg"].mean().unstack("gamma").round(1))
tblA["lazy−rich"] = (tblA[max(GAMMAS)] - tblA[min(GAMMAS)]).round(1)
print(tblA)

print("\\n" + "=" * 70)
print("B) Forward transfer err(B-start) by arch × γ (°, lower=rule reused)")
tblB = ref.groupby(["arch_disp", "gamma"])["err_B_start"].mean().unstack("gamma").round(1)
print(tblB)

print("\\n" + "=" * 70)
print("C) Effective dim #PCs(99%) at A1-end: rich vs lazy")
tblC = (ref[ref.gamma.isin([min(GAMMAS), max(GAMMAS)])]
        .groupby(["arch_disp", "gamma"])["npc99_A1"].mean().unstack("gamma").round(2))
print(tblC)

print("\\n" + "=" * 70)
print("D) Holton orthogonalisation angle (task-A vs task-B @ post-B)")
tblD = ref.groupby(["arch_disp", "gamma"])["pang_tasks_postB"].mean().unstack("gamma").round(1)
print(tblD)

print("\\n" + "=" * 70)
print("E) Rule-pull (mid-range s) rich vs lazy")
midr = ref[ref.sim_idx.isin([1, 2, 3, 4, 5]) & ref.gamma.isin([min(GAMMAS), max(GAMMAS)])]
tblE = midr.groupby(["arch_disp", "gamma"])["rs_pull"].mean().unstack("gamma").round(2)
print(tblE)
""")

md("""## 13. Paper-style figures (nb03 analogs)

Accuracy timecourse, transfer/interference in accuracy units, principal angle,
effective dimensionality, and all-architecture 3D PCA — all on **novel** stimuli.
""")

code("""archs = ARCH_ORDER
cond_idx = [SAME_IDX, NEAR_IDX, FAR_IDX]

def build_timecourse(H=REF_H, scope=REF_SCOPE, sim_idx=cond_idx, seeds=(0, 1, 2), cache=True):
    path = PROJECT_ROOT / "data" / f"overnight_tc_{REGIME}_even_H{H}_{scope}.pkl"
    if cache and path.exists():
        tc = pd.read_pickle(path)
        print(f"loaded cached timecourse: {len(tc)} rows")
        return tc
    rows = []
    for (arch, off) in ARCH_CONFIGS:
        lab = arch_label(arch, off)
        for g in GAMMAS:
            for si in sim_idx:
                for seed in seeds:
                    rid = RunConfig(arch=arch, hidden_size=H, gamma=g,
                                    similarity=SIMILARITY_GRID[si], seed=seed,
                                    off_module_policy=off, epochs_per_phase=EPOCHS,
                                    similarity_index=si, init_scope=scope,
                                    stimulus_regime=REGIME).run_id()
                    try:
                        ext = load_result(OUT_ROOT, rid)["extractions"]
                    except Exception:
                        continue
                    P, T, ge = ext["preds"], ext["targets"], ext["global_epoch"]
                    pa = np.arctan2(P[..., 1], P[..., 0]); ta = np.arctan2(T[..., 1], T[..., 0])
                    d = np.degrees(np.abs(np.arctan2(np.sin(pa - ta), np.cos(pa - ta))))
                    acc = (d < ACC_THRESH_DEG).mean(axis=1)
                    o = np.argsort(ge)
                    for gg, aa in zip(ge[o], acc[o]):
                        rows.append({"arch": lab, "gamma": g, "sim_idx": si, "seed": seed,
                                     "global_epoch": int(gg), "acc": float(aa)})
    tc = pd.DataFrame(rows)
    if cache:
        tc.to_pickle(path)
    print(f"built timecourse: {len(tc)} rows")
    return tc

tc = build_timecourse()
dfp = core2

fig, axes = plt.subplots(len(archs), 3, figsize=(11, 2.0 * len(archs)), sharex=True, sharey=True, squeeze=False)
for r, lab in enumerate(archs):
    for c, si in enumerate(cond_idx):
        ax = axes[r, c]
        sub = tc[(tc.arch == lab) & (tc.sim_idx == si)]
        for g in GAMMAS:
            m = sub[sub.gamma == g].groupby("global_epoch")["acc"].mean().sort_index()
            ax.plot(m.index, m.values, color=G_COLOR[g], lw=1.4)
        ax.axvline(EPOCHS, ls=":", c="grey", lw=0.8); ax.axvline(2 * EPOCHS, ls=":", c="grey", lw=0.8)
        ax.set_ylim(-0.02, 1.02)
        if r == 0: ax.set_title(COND[si])
        if c == 0: ax.set_ylabel(f"{ARCH_DISPLAY[lab]}\\nacc", fontsize=8)
        if r == len(archs) - 1:
            ax.set_xticks([EPOCHS/2, 1.5*EPOCHS, 2.5*EPOCHS]); ax.set_xticklabels(["A1", "B", "A2"])
fig.legend(handles=g_handles(), title="γ", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 1a — Accuracy timecourse ({REGIME})", y=1.005)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1a_accuracy_timecourse.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""fig, axes = plt.subplots(2, len(archs), figsize=(3.0 * len(archs), 6), sharex=True, sharey="row", squeeze=False)
for col, lab in enumerate(archs):
    for row, metric in enumerate(["transfer_acc", "interf_acc"]):
        ax = axes[row, col]
        sub = dfp[dfp.arch_lab == lab]
        for g in GAMMAS:
            agg = sub[sub.gamma == g].groupby("similarity")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["similarity"], agg["mean"], yerr=agg["std"].fillna(0),
                        color=G_COLOR[g], marker="o", ms=3, capsize=2, lw=1.2)
        ax.axhline(0, c="grey", lw=0.6)
        if row == 0: ax.set_title(ARCH_DISPLAY[lab], fontsize=9)
        if col == 0:
            ax.set_ylabel("transfer\\n(switch cost)" if row == 0 else "interference\\n(acc lost)")
        if row == 1: ax.set_xlabel("similarity s")
fig.legend(handles=g_handles(), title="γ", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 1b — Transfer & interference ({REGIME})", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1b_transfer_interference.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""gammas_lazy_first = sorted(GAMMAS, reverse=True)
cond_order = [SAME_IDX, NEAR_IDX, FAR_IDX]
cond_names = [COND[i] for i in cond_order]
fig, axes = plt.subplots(1, len(gammas_lazy_first), figsize=(2.6 * len(gammas_lazy_first), 3.2), sharey=True, squeeze=False)
for c, g in enumerate(gammas_lazy_first):
    ax = axes[0, c]
    for lab in archs:
        sub = dfp[(dfp.arch_lab == lab) & (dfp.gamma == g) & (dfp.sim_idx.isin(cond_order))]
        agg = sub.groupby("sim_idx")["pang_A1B"].agg(["mean", "std"]).reindex(cond_order)
        ax.errorbar(range(3), agg["mean"], yerr=agg["std"].fillna(0),
                    color=A_COLOR[lab], marker="o", ms=4, capsize=2, lw=1.4)
    tag = " (lazy)" if g == max(GAMMAS) else (" (rich)" if g == min(GAMMAS) else "")
    ax.set_title(f"γ={g:g}{tag}", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(cond_names)
    if c == 0: ax.set_ylabel("Principal angle A1↔B (°)")
fig.legend(handles=a_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 2a — Principal angle ({REGIME})", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2a_principal_angle.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""fig, axes = plt.subplots(len(archs), len(gammas_lazy_first),
                         figsize=(2.2 * len(gammas_lazy_first), 1.7 * len(archs)), sharey=True, squeeze=False)
bw = 0.25
for r, lab in enumerate(archs):
    for c, g in enumerate(gammas_lazy_first):
        ax = axes[r, c]
        sub = dfp[(dfp.arch_lab == lab) & (dfp.gamma == g)]
        for (col, base) in [("npc99_A1", 0.0), ("npc99_B", 1.2)]:
            for j, si in enumerate(cond_order):
                vals = sub[sub.sim_idx == si][col]
                ax.bar(base + j * bw, vals.mean(), bw, yerr=vals.std(),
                       color=COND_COLOR[COND[si]], capsize=1.5, error_kw={"lw": 0.6})
        ax.set_ylim(0, CONSTANTS.n_objects + 0.5)
        ax.set_xticks([0.0 + bw, 1.2 + bw]); ax.set_xticklabels(["Post-A1", "Post-B"], fontsize=7)
        if r == 0:
            tag = " (lazy)" if g == max(GAMMAS) else (" (rich)" if g == min(GAMMAS) else "")
            ax.set_title(f"γ={g:g}{tag}", fontsize=8)
        if c == 0:
            ax.set_ylabel(f"{ARCH_DISPLAY[lab]}\\n#PCs 99%", fontsize=7)
handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLOR[n]) for n in ["Same", "Near", "Far"]]
fig.legend(handles, ["Same", "Near", "Far"], loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 2b — Effective dimensionality ({REGIME})", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2b_effective_dim.png", dpi=150, bbox_inches="tight")
plt.show()
""")

code("""GEO_SEED = 0
lazy_g, rich_g = max(GAMMAS), min(GAMMAS)
n_arch, n_sim = len(ARCH_CONFIGS), len(cond_order)
n_rows, n_cols = 2 * n_arch, n_sim

fig = plt.figure(figsize=(2.8 * n_cols, 2.0 * n_rows))
for ri, g in enumerate([lazy_g, rich_g]):
    for bi, (arch, off) in enumerate(ARCH_CONFIGS):
        row = ri * n_arch + bi
        lab = arch_label(arch, off)
        for ci, si in enumerate(cond_order):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + ci + 1, projection="3d")
            try:
                hA1 = phase_end_hidden(arch, off, g, si, "A1", seed=GEO_SEED)
                hB = phase_end_hidden(arch, off, g, si, "B", seed=GEO_SEED)
                hA2 = phase_end_hidden(arch, off, g, si, "A2", seed=GEO_SEED)
                _, qA1, qB, qA2 = ana.shared_pca_three_phase(hA1, hB, hA2, n_components=3)
                for q, ph in [(qA1, "A1"), (qB, "B"), (qA2, "A2")]:
                    loop(ax, q, PHASE_COLOR[ph])
            except Exception:
                ax.text2D(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if row == 0: ax.set_title(COND[si], fontsize=9)
            if ci == 0:
                regime_tag = "lazy" if ri == 0 else "rich"
                ax.text2D(-0.18, 0.5, f"{regime_tag} (γ={g:g})\\n{ARCH_DISPLAY[lab]}",
                          transform=ax.transAxes, rotation=90, va="center", fontsize=7)
handles = [Line2D([0], [0], color=PHASE_COLOR[p], marker="o", lw=1.2, label=f"Post-{p}") for p in ["A1", "B", "A2"]]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
fig.suptitle(f"Fig 3 — 3D PCA all architectures ({REGIME})", y=1.0)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_repr_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 14. Holton-style subspace orthogonalisation

At the fixed **post-B** state, compare task-A vs task-B stimulus subspaces.
**Fig 9a** — principal angle vs similarity (all γ). **Fig 9b** — joint PCA per γ.
""")

code("""oref = core2
fig, axes = plt.subplots(1, len(ARCH_ORDER), figsize=(2.4 * len(ARCH_ORDER), 2.8), sharex=True, sharey=True, squeeze=False)
for c, lab in enumerate(ARCH_ORDER):
    ax = axes[0, c]
    sub = oref[oref.arch_lab == lab]
    for g in GAMMAS:
        m = sub[sub.gamma == g].groupby("similarity")["pang_tasks_postB"].mean().sort_index()
        ax.plot(m.index, m.values, color=G_COLOR[g], lw=1.4, marker="o", ms=2.5)
    ax.axhline(90, ls=":", c="grey", lw=0.8)
    ax.set_ylim(-3, 95)
    ax.set_title(ARCH_DISPLAY[lab], fontsize=8)
    if c == 0:
        ax.set_ylabel("principal angle (°)\\ntask-A vs task-B @ post-B", fontsize=8)
    ax.set_xlabel("similarity", fontsize=8)
fig.legend(handles=g_handles(), title="γ", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle(f"Fig 9a — Holton orthogonalisation angle ({REGIME}; 90°=orthogonal)", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig9a_orthogonalisation_angle.png", dpi=150, bbox_inches="tight")
plt.show()

print("mean pang_tasks_postB by arch × γ:")
print(oref.groupby(["arch_lab", "gamma"])["pang_tasks_postB"].mean().round(1).unstack("gamma"))
""")

code("""HOLTON_SEED = 0

def task_hidden_postB(arch, off, g, si, H=REF_H, scope=REF_SCOPE, seed=HOLTON_SEED):
    rid = RunConfig(arch=arch, hidden_size=H, gamma=g, similarity=SIMILARITY_GRID[si],
                    seed=seed, off_module_policy=off, epochs_per_phase=EPOCHS,
                    similarity_index=si, init_scope=scope, stimulus_regime=REGIME).run_id()
    ext = load_result(OUT_ROOT, rid)["extractions"]
    ph = ext["phase"].astype(str); ep = ext["epoch"].astype(int)
    b = np.where(ph == "B")[0]; a2 = np.where(ph == "A2")[0]
    hB = ext["hidden"][int(b[np.argmax(ep[b])])]
    hA = ext["hidden"][int(a2[np.argmin(ep[a2])])]
    return hA, hB

def joint_pca(mats, k=3):
    st = np.vstack(mats); mu = st.mean(0)
    _, _, Vt = np.linalg.svd(st - mu, full_matrices=False)
    comps = Vt[:k]
    return [(m - mu) @ comps.T for m in mats]

def loop3(ax, pts, color):
    q = np.atleast_2d(pts)[:, :3]
    p = np.vstack([q, q[:1]])
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color=color, ms=2.5, lw=0.9)

def plot_fig9b(g):
    cond = [SAME_IDX, NEAR_IDX, FAR_IDX]
    reg_g = "rich" if g == min(GAMMAS) else ("lazy" if g == max(GAMMAS) else "")
    fig = plt.figure(figsize=(2.8 * 3, 2.4 * len(ARCH_ORDER)))
    for r, (arch, off) in enumerate(ARCH_CONFIGS):
        lab = arch_label(arch, off)
        for c, si in enumerate(cond):
            ax = fig.add_subplot(len(ARCH_ORDER), 3, r * 3 + c + 1, projection="3d")
            try:
                hA, hB = task_hidden_postB(arch, off, g, si)
                qA, qB = joint_pca([hA, hB], 3)
                loop3(ax, qA, "tab:orange"); loop3(ax, qB, "tab:blue")
                ang = float(ana.principal_angles(hA, hB, 2)[0])
                ax.set_title((COND[si] + "  " if r == 0 else "") + f"∠={ang:.0f}°", fontsize=8)
            except Exception:
                ax.text2D(0.5, 0.5, "missing", transform=ax.transAxes, ha="center")
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            if c == 0:
                ax.text2D(-0.25, 0.5, ARCH_DISPLAY[lab], transform=ax.transAxes,
                          rotation=90, va="center", fontsize=8)
    handles = [Line2D([0], [0], color="tab:orange", marker="o", lw=1.2, label="task-A stimuli"),
               Line2D([0], [0], color="tab:blue", marker="o", lw=1.2, label="task-B stimuli")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    tag = f", {reg_g}" if reg_g else ""
    fig.suptitle(f"Fig 9b — Holton PCA ({REGIME}{tag} γ={g:g}); ∠ = principal angle", y=1.0)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"fig9b_holton_pca_gamma{g:g}.png", dpi=150, bbox_inches="tight")
    plt.show()

for g in GAMMAS:
    plot_fig9b(g)
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT} ({len(cells)} cells)")
