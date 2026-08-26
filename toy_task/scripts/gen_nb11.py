#!/usr/bin/env python3
"""Generate notebooks/11_bandwidth_phase_a.ipynb (Phase A comms_bandwidth sweep)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "11_bandwidth_phase_a.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    })


md("""# toy_task — Phase A bandwidth analysis (nb11)

Pre-registered analysis of the **communication bandwidth** sweep on
`dense` vs `modular_shared` (shared stimuli, even angles, H=50).

Data roots (first existing wins):
- `data/runs_bandwidth_phase_a/` — full 8000-epoch grid
- `data/runs_bandwidth_phase_a_pilot/` — 500-epoch pilot

Figures: `figures/nb11/`.

**Interpretation guardrail:** a flat bandwidth result means recurrent coupling under
mod-shared wiring is not the lever — *not* that modularity is irrelevant (feature-routing
and task-routing axes remain untested).
""")

md("## 1. Setup")

code("""import sys, json, glob, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from tqdm.auto import tqdm

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import toy_task
from toy_task import (
    CONSTANTS, RunConfig, SIMILARITY_GRID, COMMS_BANDWIDTH_GRID,
    PAPER_SIM_IDX, PAPER_SIM_LABELS, SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR,
    ANGLE_EVEN, STIMULUS_SHARED,
)
from toy_task import analysis as ana
from toy_task.storage import load_result

FIG_DIR = PROJECT_ROOT / "figures" / "nb11"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ROOTS = [
    PROJECT_ROOT / "data" / "runs_bandwidth_phase_a",
    PROJECT_ROOT / "data" / "runs_bandwidth_phase_a_pilot",
]
OUT_ROOT = next((p for p in CANDIDATE_ROOTS if p.exists()), CANDIDATE_ROOTS[0])
CACHE = PROJECT_ROOT / "data" / f"bandwidth_phase_a_summary_{OUT_ROOT.name}.pkl"

ARCH_ORDER = ["dense", "modular_shared"]
ARCH_DISPLAY = {"dense": "Dense (1 module)", "modular_shared": "Mod-shared"}
BANDWIDTHS = list(COMMS_BANDWIDTH_GRID)
GAMMAS = [0.001, 0.1, 2.0]
REF_H, REF_SCOPE = 50, "all"
REGIME = STIMULUS_SHARED
COND = PAPER_SIM_LABELS
PAPER_IDX = PAPER_SIM_IDX
SAME_IDX, NEAR_IDX, FAR_IDX = SIM_IDX_SAME, SIM_IDX_NEAR, SIM_IDX_FAR
EXPECTED_RUNS = 180
ACC_THRESH_DEG = 22.5

BW_COLOR = {bw: c for bw, c in zip(BANDWIDTHS, plt.cm.plasma(np.linspace(0.1, 0.9, len(BANDWIDTHS))))}
G_COLOR = {g: c for g, c in zip(GAMMAS, plt.cm.viridis(np.linspace(0, 1, len(GAMMAS))))}
A_COLOR = {a: c for a, c in zip(ARCH_ORDER, ["#4c72b0", "#dd8452"])}
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

def bw_handles():
    return [Line2D([0], [0], color=BW_COLOR[bw], lw=2, label=f"bw={bw:g}") for bw in BANDWIDTHS]

def infer_epochs(run_ids):
    eps = []
    for rid in run_ids[:20]:
        m = re.search(r"_e(\\d+)_", rid)
        if m:
            eps.append(int(m.group(1)))
    return max(eps) if eps else None

print("toy_task", toy_task.__version__)
print("OUT_ROOT:", OUT_ROOT, "| exists:", OUT_ROOT.exists())
""")

md("""## 2. Load runs → cached summary

One row per run with phase-boundary metrics, geometry, and diagnostic probes.
Re-run with `build_summary(force=True)` after new runs land.
""")

code("""def _boundary_idx(ext):
    ph = ext["phase"].astype(str); ep = ext["epoch"].astype(int); out = {}
    for phase in ("A1", "B", "A2"):
        w = np.where(ph == phase)[0]
        out[phase + "_first"] = int(w[np.argmin(ep[w])])
        out[phase + "_last"] = int(w[np.argmax(ep[w])])
    return out

def _row(meta):
    cfg, beh, ext = meta["config"], meta["behavioral"], meta["extractions"]
    ix = _boundary_idx(ext)
    Hd, P, T = ext["hidden"], ext["preds"], ext["targets"]
    a1e, bs, be, a2s, a2e = (ix["A1_last"], ix["B_first"], ix["B_last"],
                             ix["A2_first"], ix["A2_last"])
    err = lambda i: float(np.mean(ana.angular_error_deg(P[i], T[i])))
    acc = lambda i: float(np.mean(ana.angular_error_deg(P[i], T[i]) < ACC_THRESH_DEG))
    rs = ana.rule_shift(P[a2s], T[a2s], cfg["similarity"])
    arch = cfg["arch"]
    row = {
        "arch": arch, "arch_disp": ARCH_DISPLAY[arch], "gamma": cfg["gamma"],
        "bandwidth": float(cfg.get("comms_bandwidth", 0.0)),
        "similarity": round(cfg["similarity"], 4), "sim_idx": cfg["similarity_index"],
        "H": cfg["hidden_size"], "scope": cfg["init_scope"],
        "regime": cfg["stimulus_regime"], "seed": cfg["seed"],
        "epochs": cfg["epochs_per_phase"],
        "err_A1_end": err(a1e), "err_B_start": err(bs), "err_B_end": err(be),
        "err_A2_start": err(a2s), "err_A2_end": err(a2e),
        "acc_A1_end": acc(a1e), "acc_B_start": acc(bs), "acc_B_end": acc(be),
        "acc_A2_start": acc(a2s), "acc_A2_end": acc(a2e),
        "loss_ft_Ts": beh.get("forward_transfer_Ts", np.nan),
        "loss_ft_Ts_task0": beh.get("forward_transfer_Ts_task0", np.nan),
        "loss_interf_T0": beh.get("interference_T0", np.nan),
        "rs_pull": rs["pull_fraction"],
        "pr_A1": ana.participation_ratio(Hd[a1e]),
        "npc99_A1": ana.n_components_for_variance(Hd[a1e], (0.99,))[0.99],
        "npc99_B": ana.n_components_for_variance(Hd[be], (0.99,))[0.99],
        "pang_A1B": float(ana.principal_angles(Hd[a1e], Hd[be], 2)[0]),
        "pang_tasks_postB": float(ana.principal_angles(Hd[a2s], Hd[be], 2)[0]),
        "drift_A_overB": ana.hidden_drift(Hd[a1e], Hd[a2s]),
    }
    for k, v in beh.items():
        if k.startswith("probe_module"):
            row[k] = v
    return row

REQUIRED_COLS = {"bandwidth", "interf_deg", "loss_ft_Ts_task0", "probe_module0_forward_mse"}

def build_summary(force=False):
    run_ids = sorted(os.path.basename(os.path.dirname(p))
                     for p in glob.glob(str(OUT_ROOT / "*" / "config.json")))
    if CACHE.exists() and not force:
        df = pd.read_pickle(CACHE)
        if len(df) == len(run_ids) and REQUIRED_COLS.issubset(df.columns):
            print(f"loaded cached summary: {len(df)} runs")
            return df, run_ids
        print(f"cache stale ({len(df)} cached vs {len(run_ids)} on disk) -> rebuilding")
    rows = []
    for rid in tqdm(run_ids, desc="loading runs"):
        try:
            rows.append(_row(load_result(OUT_ROOT, rid)))
        except Exception as e:
            print("skip", rid, type(e).__name__, e)
    df = pd.DataFrame(rows)
    df["interf_deg"] = df["err_A2_start"] - df["err_A1_end"]
    df["transfer_deg"] = df["err_B_start"] - df["err_A1_end"]
    df["recovery_deg"] = df["err_A2_end"]
    df["interf_acc"] = df["acc_A1_end"] - df["acc_A2_start"]
    df["transfer_acc"] = df["acc_B_start"] - df["acc_A1_end"]
    df.to_pickle(CACHE)
    print(f"built + cached {len(df)} runs -> {CACHE}")
    return df, run_ids

df, run_ids = build_summary()
EPOCHS = infer_epochs(run_ids) or int(df.epochs.mode().iloc[0] if len(df) else 500)

print(f"runs on disk: {len(run_ids)} / {EXPECTED_RUNS} expected")
print(f"epochs/phase: {EPOCHS}")
if len(run_ids) < EXPECTED_RUNS:
    print("NOTE: partial grid — figures use available runs only")
if len(df) and df.err_A1_end.mean() >= 15:
    print(f"WARNING: mean A1-end error {df.err_A1_end.mean():.1f}° — training may not have converged")
else:
    print(f"OK: A1-end error mean {df.err_A1_end.mean():.2f}°")
df.head()
""")

md("""## 3. Coverage & training sanity

Confirm which factorial cells are present and whether A1 learned.
""")

code("""cov = (df.groupby(["arch", "bandwidth", "gamma", "sim_idx"]).size()
         .unstack("sim_idx", fill_value=0))
print("coverage (runs per arch×bw×γ×sim):")
display(cov.head(20))

core = df[(df.H == REF_H) & (df.scope == REF_SCOPE) & (df.regime == REGIME)].copy()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, arch in zip(axes, ARCH_ORDER):
    sub = core[core.arch == arch]
    for g in GAMMAS:
        m = sub[sub.gamma == g].groupby("bandwidth")["err_A1_end"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m.values, marker="o", color=G_COLOR[g], label=f"γ={g:g}")
    ax.set_xlabel("comms bandwidth"); ax.set_ylabel("A1-end angular error (°)")
    ax.set_title(ARCH_DISPLAY[arch]); ax.axhline(2.0, ls=":", c="grey", lw=0.8)
axes[1].legend(fontsize=8, loc="upper right")
fig.suptitle(f"Fig 0 — Training sanity ({EPOCHS} epochs/phase)", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig0_training_sanity.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 4. Primary — interference vs bandwidth

Pre-registered primary metric. Lines per γ; columns = architecture; rows = similarity.
""")

code("""fig, axes = plt.subplots(len(PAPER_IDX), len(ARCH_ORDER),
                         figsize=(4.2 * len(ARCH_ORDER), 3.2 * len(PAPER_IDX)),
                         sharex=True, squeeze=False)
for row, si in enumerate(PAPER_IDX):
    for col, arch in enumerate(ARCH_ORDER):
        ax = axes[row, col]
        sub = core[(core.arch == arch) & (core.sim_idx == si)]
        for g in GAMMAS:
            m = sub[sub.gamma == g].groupby("bandwidth")["interf_deg"].agg(["mean", "std"])
            m = m.reindex(BANDWIDTHS)
            ax.errorbar(BANDWIDTHS, m["mean"], yerr=m["std"].fillna(0),
                        color=G_COLOR[g], marker="o", capsize=3, lw=1.4)
        ax.axhline(0, c="grey", lw=0.6)
        if row == 0:
            ax.set_title(ARCH_DISPLAY[arch])
        if col == 0:
            ax.set_ylabel(f"{COND[si]}\\ninterference (°)")
        if row == len(PAPER_IDX) - 1:
            ax.set_xlabel("comms bandwidth")
fig.legend(handles=g_handles(), title="γ", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle("Fig 1 — Interference vs bandwidth (primary)", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_interference_vs_bandwidth.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 5. Transfer & geometry vs bandwidth

Switch cost (`transfer_deg`), subspace reorientation (`pang_A1B`), and effective
dimensionality (`npc99_A1`) — averaged over seeds per cell.
""")

code("""metrics = [
    ("transfer_deg", "switch cost (°)"),
    ("pang_A1B", "principal angle A1↔B (°)"),
    ("npc99_A1", "#PCs 99% (A1-end)"),
]
fig, axes = plt.subplots(len(metrics), len(ARCH_ORDER),
                         figsize=(4.5 * len(ARCH_ORDER), 3.0 * len(metrics)), sharex=True, squeeze=False)
for row, (metric, ylab) in enumerate(metrics):
    for col, arch in enumerate(ARCH_ORDER):
        ax = axes[row, col]
        sub = core[core.arch == arch]
        for g in GAMMAS:
            m = sub[sub.gamma == g].groupby("bandwidth")[metric].mean().reindex(BANDWIDTHS)
            ax.plot(BANDWIDTHS, m.values, marker="o", color=G_COLOR[g], lw=1.4)
        if row == 0:
            ax.set_title(ARCH_DISPLAY[arch])
        if col == 0:
            ax.set_ylabel(ylab)
        if row == len(metrics) - 1:
            ax.set_xlabel("comms bandwidth")
fig.legend(handles=g_handles(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.suptitle("Fig 2 — Transfer & geometry vs bandwidth", y=1.01)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_transfer_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 6. Heatmaps — γ × bandwidth (mod-shared, reference similarities)

Collapse seeds; facet by Same / Near / Far.
""")

code("""arch = "modular_shared"
fig, axes = plt.subplots(1, len(PAPER_IDX), figsize=(3.4 * len(PAPER_IDX), 3.2), squeeze=False)
for c, si in enumerate(PAPER_IDX):
    ax = axes[0, c]
    sub = core[(core.arch == arch) & (core.sim_idx == si)]
    M = (sub.groupby(["gamma", "bandwidth"])["interf_deg"].mean()
         .unstack("bandwidth").reindex(index=GAMMAS, columns=BANDWIDTHS))
    im = ax.imshow(M.values, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(BANDWIDTHS))); ax.set_xticklabels([f"{b:g}" for b in BANDWIDTHS], fontsize=8)
    ax.set_yticks(range(len(GAMMAS))); ax.set_yticklabels([f"{g:g}" for g in GAMMAS], fontsize=8)
    ax.set_title(COND[si]); ax.set_xlabel("bandwidth")
    if c == 0:
        ax.set_ylabel("γ (rich→lazy)")
fig.colorbar(im, ax=axes, shrink=0.85, label="interference (°)")
fig.suptitle("Fig 3 — Interference heatmap (mod-shared)", y=1.02)
fig.savefig(FIG_DIR / "fig3_interference_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 7. Diagnostic probes

`forward_transfer_Ts` uses `task_id=1` (confounded for task-routed). Phase A grid
uses dense/mod-shared only, but we still plot **`forward_transfer_Ts_task0`** and
per-module linear probes fit on A1-end (A objects) and tested on B (T(s)).
""")

code("""probe_cols = [c for c in df.columns if c.startswith("probe_module")]
has_probes = len(probe_cols) > 0
print("probe columns:", probe_cols[:6], "..." if len(probe_cols) > 6 else "")

if has_probes:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    sub = core[core.arch == "modular_shared"]
    # pipeline forward transfer (MSE loss, degrees on second row for angular err)
    ax = axes[0, 0]
    for g in GAMMAS:
        m = sub[sub.gamma == g].groupby("bandwidth")["loss_ft_Ts"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m.values, marker="o", color=G_COLOR[g])
    ax.set_title("forward_transfer_Ts (task_id=1)"); ax.set_xlabel("bandwidth"); ax.set_ylabel("MSE")
    ax = axes[0, 1]
    for g in GAMMAS:
        m = sub[sub.gamma == g].groupby("bandwidth")["loss_ft_Ts_task0"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m.values, marker="o", color=G_COLOR[g])
    ax.set_title("forward_transfer_Ts_task0"); ax.set_xlabel("bandwidth"); ax.set_ylabel("MSE")
    ax = axes[1, 0]
    for g in GAMMAS:
        m = sub[sub.gamma == g].groupby("bandwidth")["probe_module0_forward_mse"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m.values, marker="o", color=G_COLOR[g])
    ax.set_title("probe_module0_forward_mse"); ax.set_xlabel("bandwidth"); ax.set_ylabel("MSE")
    ax = axes[1, 1]
    for g in GAMMAS:
        m0 = sub[sub.gamma == g].groupby("bandwidth")["probe_module0_rule_mse"].mean().reindex(BANDWIDTHS)
        m1 = sub[sub.gamma == g].groupby("bandwidth")["probe_module1_rule_mse"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m0.values, marker="o", color=G_COLOR[g], ls="-")
        ax.plot(BANDWIDTHS, m1.values, marker="s", color=G_COLOR[g], ls="--")
    ax.set_title("probe rule_mse (solid=m0, dash=m1)"); ax.set_xlabel("bandwidth")
    fig.legend(handles=g_handles(), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Fig 4 — Diagnostic probes (mod-shared)", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_probes.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No probe columns yet — re-run after pilot completes")
""")

md("""## 8. Dense vs mod-shared at matched bandwidth

Does mod-shared diverge from dense as bandwidth increases? (Dense ignores bandwidth
in the model but run_ids encode the swept level for factorial balance.)
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for ax, si in zip(axes, PAPER_IDX):
    for arch, ls in [("dense", "--"), ("modular_shared", "-")]:
        sub = core[(core.arch == arch) & (core.sim_idx == si)]
        m = sub.groupby("bandwidth")["interf_deg"].mean().reindex(BANDWIDTHS)
        ax.plot(BANDWIDTHS, m.values, marker="o", ls=ls, color=A_COLOR[arch], label=ARCH_DISPLAY[arch])
    ax.set_title(COND[si]); ax.set_xlabel("bandwidth"); ax.axhline(0, c="grey", lw=0.5)
    if ax is axes[0]:
        ax.set_ylabel("interference (°)")
axes[0].legend(fontsize=8)
fig.suptitle("Fig 5 — Dense vs mod-shared (γ pooled)", y=1.02)
fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_dense_vs_modshared.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""## 9. Pre-registration evaluator (§2 of PHASE_A_BANDWIDTH_PLAN)

Classifies outcome **A** (monotone gradient), **B** (flat), **C** (binary switch),
or **D** (inconclusive). Criteria applied to `modular_shared` primary metrics.
""")

code("""PRIMARY_METRICS = ["interf_deg", "pang_A1B", "npc99_A1", "transfer_deg"]
RHO_THRESH = 0.6
BWS = np.array(BANDWIDTHS, dtype=float)

def spearman_across_bw(sub, metric):
    g = sub.groupby("bandwidth")[metric].mean().reindex(BANDWIDTHS)
    if g.isna().any() or len(g) < 3:
        return np.nan, np.nan
    rho, p = spearmanr(BWS, g.values)
    return float(rho), float(p)

def classify_slice(sub, metric):
    rho, p = spearman_across_bw(sub, metric)
    if np.isnan(rho):
        return "D", rho, p
    if abs(rho) >= RHO_THRESH:
        return "A", rho, p
    vals = sub.groupby("bandwidth")[metric].mean().reindex(BANDWIDTHS)
    mid = vals.iloc[1:-1].mean()
    lo, hi = vals.iloc[0], vals.iloc[-1]
    if abs(lo - hi) > 2 * abs(mid - lo) and abs(mid - lo) < 0.25 * abs(hi - lo):
        return "C", rho, p
    if abs(lo - hi) < 0.5 * max(abs(vals.std()), 1e-6) and abs(rho) < 0.3:
        return "B", rho, p
    return "D", rho, p

rows = []
ms = core[core.arch == "modular_shared"]
for metric in PRIMARY_METRICS:
    for si in PAPER_IDX:
        for g in GAMMAS:
            sub = ms[(ms.sim_idx == si) & (ms.gamma == g)]
            outcome, rho, p = classify_slice(sub, metric)
            rows.append({"metric": metric, "sim": COND[si], "gamma": g,
                         "outcome": outcome, "spearman_rho": rho, "p_value": p})
eval_df = pd.DataFrame(rows)
print("Per-slice classification (mod-shared):")
display(eval_df.round(3))

summary = (eval_df.groupby("metric")["outcome"]
           .value_counts().unstack(fill_value=0))
print("\\nOutcome counts by metric:")
display(summary)

# Global outcome: A if any metric has ≥2 agreeing-A slices (same sign rho across ≥2 γ or ≥2 sims)
def global_outcome(edf):
    a_hits = []
    for metric in PRIMARY_METRICS:
        msub = edf[edf.metric == metric]
        strong = msub[msub.outcome == "A"]
        if len(strong) >= 2:
            signs = np.sign(strong.spearman_rho)
            if len(set(signs)) == 1:
                a_hits.append(metric)
    if a_hits:
        return "A", f"monotone gradient in: {', '.join(a_hits)}"
    b_frac = (edf.outcome == "B").mean()
    if b_frac >= 0.6:
        return "B", "flat across bandwidth — recurrent coupling not the lever under mod-shared wiring"
    c_frac = (edf.outcome == "C").mean()
    if c_frac >= 0.4:
        return "C", "threshold / regime-switch pattern"
    return "D", "inconclusive — increase seeds or epochs before Phase B"

outcome, rationale = global_outcome(eval_df)
print("\\n" + "=" * 60)
print(f"PHASE A OUTCOME: {outcome}")
print(rationale)
print("=" * 60)
print("\\nReminder: flat bandwidth ≠ modularity irrelevant (feature-routing untested).")
""")

md("""## 10. Quantitative digest
""")

code("""pd.set_option("display.width", 120)
ms = core[core.arch == "modular_shared"]

print("A) Interference (°) by bandwidth × γ — mod-shared, Near")
tbl = (ms[ms.sim_idx == NEAR_IDX]
       .groupby(["bandwidth", "gamma"])["interf_deg"].mean().unstack("gamma").round(1))
print(tbl)

print("\\nB) Effective dim #PCs(99%) A1-end — rich vs lazy, pooled sims")
tbl2 = (ms[ms.gamma.isin([min(GAMMAS), max(GAMMAS)])]
        .groupby(["bandwidth", "gamma"])["npc99_A1"].mean().unstack("gamma").round(2))
print(tbl2)

print("\\nC) Forward transfer err(B-start) — mod-shared vs dense @ bw=0")
tbl3 = (core[(core.bandwidth == 0) & (core.sim_idx == NEAR_IDX)]
        .groupby(["arch_disp", "gamma"])["err_B_start"].mean().unstack("gamma").round(1))
print(tbl3)

print("\\nD) Probe forward MSE module 0 (if present)")
if "probe_module0_forward_mse" in ms.columns:
    print(ms.groupby(["bandwidth", "gamma"])["probe_module0_forward_mse"].mean().unstack("gamma").round(4))
""")

md("""## 11. Takeaways

Edit after reviewing figures. The pre-registration cell above prints the formal outcome.
""")

code("""print(f"Runs analysed: {len(df)} / {EXPECTED_RUNS}")
print(f"Data: {OUT_ROOT.name} | epochs/phase: {EPOCHS}")
print(f"Figures saved to: {FIG_DIR}")
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
