#!/usr/bin/env python3
"""Generate notebooks/09_task_diagnosis.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "09_task_diagnosis.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [text],
        }
    )


md("""# toy_task — Task design diagnostics (nb09)

**Goal:** Decide whether the next implementation step should be

1. **Evenly spaced object angles** (patch-faithful ring, 45° min separation), or
2. **Summer/winter seasons** (dual-output relational structure),

before committing to a full grid.

**Signals tested (verifiable):**

| Test | What it isolates |
|------|------------------|
| Angle geometry | Object separability: random `atan2(z)` vs even ring |
| Oracle linear probe | Is the rule *statistically learnable* from `x` without an RNN? |
| Forward transfer after A1 | Does the RNN infer global `+s` on novel B without B training? |
| Full A1→B→A2 | End-state learning + interference under both angle layouts |
| min-sep correlation | Is poor forward transfer driven by clustered random angles? |

**Outputs:**

- Per-run cache: `data/runs_nb09/<run_id>/metrics.json` (resumable, like overnight grid)
- Aggregated summary: `data/nb09_diagnostics.pkl`
- Figures: `figures/nb09/`

> Does **not** touch `data/runs_overnight/`.
""")

md("## 1. Setup")
code("""import sys, pickle, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import toy_task
from toy_task import (
    CONSTANTS, RunConfig, SIMILARITY_GRID,
    PAPER_SIM_IDX, PAPER_SIM_LABELS, SIM_IDX_NEAR,
    ANGLE_RANDOM, ANGLE_EVEN, STIMULUS_NOVEL,
)
from toy_task.environment import Environment
from toy_task import diagnostics as diag

FIG_DIR = PROJECT_ROOT / "figures" / "nb09"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_ROOT = DATA_DIR / "runs_nb09"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "nb09_diagnostics.pkl"
FORCE_RECOMPUTE = False   # set True to ignore per-run cache

# --- Verifiable budget (not overnight, but enough for stable readouts) ---
EPOCHS = 1500          # per phase
H = 50
SEEDS = list(range(3))
GAMMAS = [0.001, 1.0, 2.0]
SIM_NEAR = SIMILARITY_GRID[SIM_IDX_NEAR]
SIM_IDX_FULL = PAPER_SIM_IDX
ARCH_PRIMARY = "dense"
ARCH_SECONDARY = "modular_shared"

print(f"toy_task {toy_task.__version__}")
print(f"epochs/phase={EPOCHS}  H={H}  seeds={SEEDS}")
print(f"Near similarity s={SIM_NEAR:.4f} rad ({np.degrees(SIM_NEAR):.1f}°)")
print(f"run cache -> {RUNS_ROOT}")
print(f"figures -> {FIG_DIR}")
""")

md("""## 2. Angle geometry — random vs even

The patch spec uses equally spaced targets (min gap 45° for N=8).
Current default draws `theta_i = atan2(z_{i,1}, z_{i,0})` which often clusters objects.
""")
code("""rng_seeds = list(range(200))
random_mins = [Environment.from_seed(s, stimulus_regime=STIMULUS_NOVEL, angle_mode=ANGLE_RANDOM).min_pairwise_sep_deg("A") for s in rng_seeds]
even_mins = [Environment.from_seed(s, stimulus_regime=STIMULUS_NOVEL, angle_mode=ANGLE_EVEN).min_pairwise_sep_deg("A") for s in rng_seeds]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(random_mins, bins=30, alpha=0.7, label=f"random (median {np.median(random_mins):.1f}°)")
ax.hist(even_mins, bins=15, alpha=0.7, label=f"even (all {even_mins[0]:.1f}°)")
ax.axvline(45, color="k", ls="--", lw=1, label="patch spec 45°")
ax.set_xlabel("Min pairwise base-angle separation (deg)")
ax.set_ylabel("Count over seeds")
ax.set_title("Object angular separability — task A")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig01_angle_separability.png", dpi=150)
plt.show()
print(f"random: median min-sep = {np.median(random_mins):.2f}°  q10 = {np.percentile(random_mins,10):.2f}°")
print(f"even:   fixed min-sep = {even_mins[0]:.2f}°")
""")

md("""## 3. Oracle linear probe — is the rule learnable from `x`?

Train `x_A → y_A` with ridge regression; evaluate on `y_B` (rule `+s`) **without**
any neural network. If oracle forward MSE is already high, the observation map
is the bottleneck (not seasons / not RNN capacity).
""")
code("""oracle_rows = []
for angle_mode in (ANGLE_RANDOM, ANGLE_EVEN):
    for seed in SEEDS:
        env = diag.build_env(seed, angle_mode=angle_mode)
        oa, ob = diag.oracle_linear_forward(env, SIM_NEAR)
        oracle_rows.append({
            "angle_mode": angle_mode, "seed": seed,
            "oracle_a_mse": oa, "oracle_forward_mse": ob,
            "min_sep_A_deg": env.min_pairwise_sep_deg("A"),
        })
oracle_df = pd.DataFrame(oracle_rows)
display(oracle_df.groupby("angle_mode")[["oracle_a_mse", "oracle_forward_mse"]].agg(["mean", "std"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col, title in zip(axes, ["oracle_a_mse", "oracle_forward_mse"], ["Fit A", "Forward B (oracle)"]):
    data = [oracle_df.loc[oracle_df.angle_mode == m, col].values for m in (ANGLE_RANDOM, ANGLE_EVEN)]
    ax.boxplot(data, labels=[ANGLE_RANDOM, ANGLE_EVEN])
    ax.set_title(title)
    ax.set_ylabel("MSE")
fig.suptitle(f"Oracle linear probe — Near s={np.degrees(SIM_NEAR):.0f}°")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig02_oracle_probe.png", dpi=150)
plt.show()
""")

md("""## 4. RNN forward transfer after A1 only

**Key diagnostic:** after training task A, measure clean MSE on novel B at `T(s)`
*before* any B updates. Low loss ⇒ rule inferred; high loss ⇒ memorisation on A only.

Compare `angle_mode` random vs even (dense, γ ladder, 3 seeds).
""")
code("""a1_configs = [
    RunConfig(
        arch=ARCH_PRIMARY, hidden_size=H, gamma=g, similarity=SIM_NEAR,
        seed=s, epochs_per_phase=EPOCHS, stimulus_regime=STIMULUS_NOVEL,
        similarity_index=SIM_IDX_NEAR,
    )
    for s in SEEDS for g in GAMMAS
]
t0 = time.time()
a1_rows = diag.run_grid(
    a1_configs, [ANGLE_RANDOM, ANGLE_EVEN], protocol="a1",
    out_root=RUNS_ROOT, force=FORCE_RECOMPUTE,
)
a1_df = diag.metrics_to_frame(a1_rows)
print(f"A1-only grid done in {time.time()-t0:.0f}s  ({len(a1_df)} runs)")

summ_a1 = a1_df.groupby(["angle_mode", "gamma"]).agg({
    "forward_transfer_Ts": ["mean", "std"],
    "a1_end_T0": ["mean", "std"],
    "hidden_err_A_deg": ["mean", "std"],
    "oracle_forward_mse": ["mean"],
}).round(4)
display(summ_a1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
metrics = [
    ("forward_transfer_Ts", "Forward transfer (↓ better)"),
    ("a1_end_T0", "A1 end loss T(0) (↓ better)"),
    ("hidden_err_A_deg", "Hidden angle error on A (↓ better)"),
]
for ax, (col, title) in zip(axes, metrics):
    pivot = a1_df.groupby(["gamma", "angle_mode"])[col].mean().unstack()
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_title(title)
    ax.set_xlabel("γ")
    ax.legend(title="angle_mode")
fig.suptitle(f"After A1 only — {ARCH_PRIMARY} H={H} Near")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig03_a1_forward_transfer.png", dpi=150)
plt.show()

score_fwd = diag.score_angle_mode_comparison(a1_df, "forward_transfer_Ts")
print("Forward-transfer score:", score_fwd["even_minus_random"], "(negative ⇒ even helps)")
""")

md("""## 5. min-sep correlation (random angles only)

If forward transfer correlates with `min_sep_A_deg`, clustered random angles are
a measurable culprit — fixing even spacing should help systematically.
""")
code("""rand_a1 = a1_df[a1_df.angle_mode == ANGLE_RANDOM].copy()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, g in zip(axes, GAMMAS):
    sub = rand_a1[rand_a1.gamma == g]
    ax.scatter(sub.min_sep_A_deg, sub.forward_transfer_Ts, alpha=0.8)
    if len(sub) > 2:
        r = np.corrcoef(sub.min_sep_A_deg, sub.forward_transfer_Ts)[0, 1]
        ax.set_title(f"γ={g}  r={r:.2f}")
    ax.set_xlabel("min sep A (deg)")
    ax.set_ylabel("forward transfer MSE")
fig.suptitle("Random angles: separability vs forward transfer")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig04_minsep_correlation.png", dpi=150)
plt.show()
""")

md("""## 6. Full A1→B→A2 protocol (Near, subset)

End-state losses and interference — does even spacing also improve behaviour after
full continual learning, or only forward transfer?

*(Paper Same/Near/Far triple is deferred until after this decision; Near is the
most informative similarity level for rule conflict.)*
""")
code("""full_configs = [
    RunConfig(
        arch=ARCH_PRIMARY, hidden_size=H, gamma=g,
        similarity=SIM_NEAR, seed=s,
        epochs_per_phase=EPOCHS, stimulus_regime=STIMULUS_NOVEL,
        similarity_index=SIM_IDX_NEAR,
    )
    for s in SEEDS for g in GAMMAS
]
# modular shared sanity (Near, same seeds)
full_configs += [
    RunConfig(
        arch=ARCH_SECONDARY, hidden_size=H, gamma=g,
        similarity=SIM_NEAR, seed=s,
        epochs_per_phase=EPOCHS, stimulus_regime=STIMULUS_NOVEL,
        similarity_index=SIM_IDX_NEAR,
    )
    for s in SEEDS for g in GAMMAS
]

t0 = time.time()
full_rows = diag.run_grid(
    full_configs, [ANGLE_RANDOM, ANGLE_EVEN], protocol="full",
    out_root=RUNS_ROOT, force=FORCE_RECOMPUTE,
)
full_df = diag.metrics_to_frame(full_rows)
print(f"Full protocol done in {time.time()-t0:.0f}s  ({len(full_df)} runs)")

dense_full = full_df[full_df.arch == ARCH_PRIMARY]
pivot = dense_full.groupby(["angle_mode", "gamma"]).agg({
    "interference_T0": "mean",
    "a2_end_T0": "mean",
    "forward_transfer_Ts": "mean",
}).round(4)
display(pivot)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(GAMMAS))
w = 0.35
for i, am in enumerate((ANGLE_RANDOM, ANGLE_EVEN)):
    sub = dense_full[(dense_full.angle_mode == am) & (dense_full.similarity == SIM_NEAR)]
    means = [sub[sub.gamma == g].interference_T0.mean() for g in GAMMAS]
    ax.bar(x + (i - 0.5) * w, means, width=w, label=am)
ax.set_xticks(x)
ax.set_xticklabels([f"γ={g}" for g in GAMMAS])
ax.set_ylabel("Interference T(0) after B (↓ better)")
ax.set_title(f"Near — full protocol {ARCH_PRIMARY}")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_full_interference.png", dpi=150)
plt.show()

mod_full = full_df[full_df.arch == ARCH_SECONDARY]
if len(mod_full):
    display(mod_full.groupby(["angle_mode", "gamma"])["forward_transfer_Ts"].agg(["mean", "std"]).round(4))
""")

md("""## 7. Decision synthesis

**Rubrics:**

| Criterion | Favors **even angles** | Favors **seasons next** |
|-----------|-------------------------|-------------------------|
| Oracle forward ≪ 1, RNN forward ≫ oracle | RNN bottleneck, not observation map | — |
| Even ≪ random on forward transfer | ✅ fix angles first | seasons can wait |
| Even ≈ random on forward transfer | angles not root cause | ✅ try seasons |
| min-sep correlation \|r\| > 0.4 | ✅ clustering matters | — |
| Full-protocol interference still flat with even | angles insufficient alone | ✅ seasons / A2 asymmetry |

*Seasons are not implemented in this notebook — if even angles fix forward transfer
but modularity×γ effects stay flat, seasons remain the next hypothesis.*
""")
code("""def decision_report(a1_df, full_df, oracle_df):
    lines = []
    # 1 Oracle
    o_rand = oracle_df.loc[oracle_df.angle_mode == ANGLE_RANDOM, "oracle_forward_mse"].mean()
    o_even = oracle_df.loc[oracle_df.angle_mode == ANGLE_EVEN, "oracle_forward_mse"].mean()
    rnn_rand = a1_df.loc[a1_df.angle_mode == ANGLE_RANDOM, "forward_transfer_Ts"].mean()
    rnn_even = a1_df.loc[a1_df.angle_mode == ANGLE_EVEN, "forward_transfer_Ts"].mean()
    lines.append(f"Oracle forward MSE: random={o_rand:.4f}  even={o_even:.4f}")
    lines.append(f"RNN forward MSE (A1): random={rnn_rand:.4f}  even={rnn_even:.4f}  Δ(even-rand)={rnn_even-rnn_rand:+.4f}")
    # 2 Correlation
    sub = a1_df[a1_df.angle_mode == ANGLE_RANDOM]
    r_vals = []
    for g in GAMMAS:
        s = sub[sub.gamma == g]
        if len(s) > 2:
            r_vals.append(np.corrcoef(s.min_sep_A_deg, s.forward_transfer_Ts)[0, 1])
    r_mean = float(np.mean(r_vals)) if r_vals else float("nan")
    lines.append(f"Mean corr(min_sep, forward) over γ: r={r_mean:.3f}")
    # 3 Full protocol near dense
    d = full_df[(full_df.arch == ARCH_PRIMARY) & (full_df.similarity == SIM_NEAR)]
    if len(d):
        intf_r = d.loc[d.angle_mode == ANGLE_RANDOM, "interference_T0"].mean()
        intf_e = d.loc[d.angle_mode == ANGLE_EVEN, "interference_T0"].mean()
        lines.append(f"Interference (Near, full): random={intf_r:.4f}  even={intf_e:.4f}")
    # Verdict
    votes_even = 0
    if rnn_even < rnn_rand - 0.02:
        votes_even += 1
        lines.append("✓ Forward transfer improves with even angles")
    else:
        lines.append("✗ Forward transfer similar (even does not clearly help)")
    if abs(r_mean) > 0.35:
        votes_even += 1
        lines.append("✓ min-sep correlates with forward transfer on random layout")
    if o_rand < 0.15 and rnn_rand > 0.25:
        lines.append("→ Oracle generalises but RNN does not: rule-learning / training issue")
    if votes_even >= 2:
        lines.append("\\n**RECOMMENDATION: implement even-spaced angles first, re-run nb08-style grid.**")
    elif rnn_even >= rnn_rand - 0.02 and o_rand < 0.2:
        lines.append("\\n**RECOMMENDATION: angle layout is not the bottleneck; prioritize seasons (+ fixed α).**")
    else:
        lines.append("\\n**RECOMMENDATION: mixed signal — run even angles AND a minimal seasons pilot.**")
    return "\\n".join(lines)

report = decision_report(a1_df, full_df, oracle_df)
print(report)

payload = {
    "epochs_per_phase": EPOCHS,
    "runs_root": str(RUNS_ROOT),
    "oracle_df": oracle_df,
    "a1_df": a1_df,
    "full_df": full_df,
    "report": report,
}
with open(CACHE, "wb") as f:
    pickle.dump(payload, f)
print(f"\\nSummary cached -> {CACHE}")
print(f"Per-run artifacts -> {RUNS_ROOT} ({len(list(RUNS_ROOT.iterdir()))} dirs)")

with open(FIG_DIR / "decision_report.txt", "w") as f:
    f.write(report)
print(f"Report -> {FIG_DIR / 'decision_report.txt'}")
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
OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}")
