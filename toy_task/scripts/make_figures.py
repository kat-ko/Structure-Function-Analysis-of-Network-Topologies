"""Aggregate saved runs into the core figure set (Part IV Section 18).

Loads every run under ``--runs``, builds a tidy behavioral table (forward transfer,
interference, effective dimensionality) and renders summary plots comparing
architectures across the similarity grid. PCA trajectories / RSA per run are left to
the notebooks; this script produces the headline behavioral + dimensionality views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toy_task.storage import load_result
from toy_task.analysis import participation_ratio


def _arch_label(meta_cfg: dict) -> str:
    arch = meta_cfg["arch"]
    if arch == "modular_task_routed":
        return f"{arch}/{meta_cfg['off_module_policy']}"
    return arch


def collect(runs_root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(runs_root.iterdir()):
        if not (d / "config.json").exists():
            continue
        meta = load_result(runs_root, d.name)
        cfg = meta["config"]
        beh = meta["behavioral"]
        ext = meta["extractions"]
        # End-of-A1 effective dimensionality from the last A1 extraction.
        pr_a1 = np.nan
        if "phase" in ext:
            phases = ext["phase"].astype(str)
            a1_idx = np.where(phases == "A1")[0]
            if len(a1_idx):
                pr_a1 = participation_ratio(ext["hidden"][a1_idx[-1]])
        rows.append(
            {
                "run_id": cfg["run_id"],
                "arch": _arch_label(cfg),
                "hidden_size": cfg["hidden_size"],
                "gamma": cfg["gamma"],
                "similarity": cfg["similarity"],
                "seed": cfg["seed"],
                "forward_transfer": beh.get("forward_transfer_Ts", np.nan),
                "interference": beh.get("interference_T0", np.nan),
                "a2_end_T0": beh.get("a2_end_T0", np.nan),
                "pr_a1": pr_a1,
            }
        )
    return pd.DataFrame(rows)


def _lineplot(ax, df, ycol, title):
    for arch, g in df.groupby("arch"):
        agg = g.groupby("similarity")[ycol].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["similarity"], agg["mean"], yerr=agg["std"],
                    marker="o", capsize=3, label=arch)
    ax.set_xlabel("task similarity s")
    ax.set_ylabel(ycol)
    ax.set_title(title)


def main() -> None:
    p = argparse.ArgumentParser(description="toy_task figure aggregation")
    p.add_argument("--runs", default="data/runs")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    runs_root = Path(args.runs)
    df = collect(runs_root)
    if df.empty:
        print(f"No runs found under {runs_root}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "behavioral_summary.csv", index=False)
    print(f"Loaded {len(df)} runs; wrote {out/'behavioral_summary.csv'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _lineplot(axes[0], df, "forward_transfer", "Forward transfer vs similarity")
    _lineplot(axes[1], df, "interference", "Interference vs similarity")
    _lineplot(axes[2], df, "pr_a1", "Effective dim (PR) end of A1")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "behavioral_summary.png", dpi=120)
    print("wrote", out / "behavioral_summary.png")


if __name__ == "__main__":
    main()
