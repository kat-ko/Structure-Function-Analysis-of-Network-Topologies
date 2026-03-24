"""
Reproduce init-scale geometry notebook pipelines and print markdown-friendly summaries.
Run from repo: python a1b2_modular/scripts/summarize_init_scale_geometry_notebooks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INIT_SCALE_ORDER = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
SPARSITY_ORDER = ["no_comms", "0.1", "0.3", "0.5", "0.7", "0.9", "1.0", "single_module"]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    root = here.parent.parent
    if not (root / "a1b2").exists():
        raise RuntimeError(f"Expected a1b2 under {root}")
    return root


def _select_rows(settings: dict, routing: str) -> list[dict]:
    TARGET_ARCH = "two_module_rnn"
    TARGET_SINGLE_ARCH = "single_module_rnn"
    TARGET_DIM_HIDDEN = 25
    TARGET_NB_STEPS = 2
    TARGET_COMMON_READOUT = True
    TARGET_COMMON_INPUT = False

    def ok(c):
        arch = c.get("arch")
        if arch == TARGET_ARCH:
            if c.get("dim_hidden") != TARGET_DIM_HIDDEN:
                return False
            if c.get("nb_steps", 1) != TARGET_NB_STEPS:
                return False
            if c.get("input_routing", "shared") != routing:
                return False
            if c.get("common_readout", True) is not TARGET_COMMON_READOUT:
                return False
            if c.get("common_input", False) is not TARGET_COMMON_INPUT:
                return False
            return True
        if arch == TARGET_SINGLE_ARCH:
            if c.get("dim_hidden") != TARGET_DIM_HIDDEN:
                return False
            if c.get("nb_steps", 1) != TARGET_NB_STEPS:
                return False
            return True
        return False

    root = _project_root()
    sys.path.insert(0, str(root))
    from a1b2.utils.run_config import build_run_id

    sim_folder = root / "data" / "simulations"
    rows = []
    for c in settings["conditions"]:
        if not ok(c):
            continue
        init_scale = c.get("init_scale", None)
        if init_scale is None:
            init_scale = 1.0
        run_id = build_run_id(c)
        path = sim_folder / run_id
        if c.get("arch") == TARGET_SINGLE_ARCH:
            sp_label = "single_module"
            sp_val = np.nan
        else:
            sp = c.get("sparsity", 1.0)
            if sp == 0 or "no_comms" in c.get("name", ""):
                sp_label = "no_comms"
            else:
                sp_float = float(sp)
                sp_label = "1.0" if np.isclose(sp_float, 1.0) else str(sp_float)
            sp_val = float(sp)
        rows.append(
            {
                "run_id": run_id,
                "init_scale": float(init_scale),
                "sparsity": sp_val,
                "sparsity_label": sp_label,
                "path_exists": path.exists(),
                "path": str(path),
            }
        )
    return rows


def _load_all_ann_data(valid_df: pd.DataFrame, ann):
    all_ann_data = {}
    for _, row in valid_df.iterrows():
        run_id = row["run_id"]
        path = Path(row["path"])
        ann_data = ann.load_ann_data(str(path), load_rnn_extra=True)
        if (
            len(ann_data.get("same", [])) > 0
            and len(ann_data.get("near", [])) > 0
            and len(ann_data.get("far", [])) > 0
        ):
            all_ann_data[run_id] = {
                "data": ann_data,
                "init_scale": row["init_scale"],
                "sparsity": row["sparsity"],
                "sparsity_label": row["sparsity_label"],
            }
    return all_ann_data


def _build_frames(all_ann_data: dict, ann):
    transfer_records = []
    accuracy_records = []
    rep_records = []
    angles_records = []

    for run_id, bundle in all_ann_data.items():
        ann_data = bundle["data"]
        sparsity = bundle["sparsity"]
        sp_label = bundle["sparsity_label"]
        init_scale = bundle["init_scale"]

        adf = ann.get_principal_angles(ann_data)
        adf = adf.copy()
        adf["run_id"] = run_id
        adf["init_scale"] = init_scale
        adf["sparsity"] = sparsity
        adf["sparsity_label"] = sp_label
        angles_records.append(adf)

        for cond_name, entries in ann_data.items():
            for entry in entries:
                pid = str(entry["participant"])
                acc = np.asarray(entry["accuracy"])
                if acc.ndim == 3:
                    cond_idx = (
                        ["same", "near", "far"].index(cond_name)
                        if cond_name in ["same", "near", "far"]
                        else 0
                    )
                    acc_cond = acc[cond_idx]
                else:
                    acc_cond = acc
                if acc_cond.ndim >= 2 and acc_cond.shape[0] >= 2:
                    A1_acc = (
                        acc_cond[0, 1::2]
                        if acc_cond.ndim == 2
                        else np.asarray(acc_cond[0]).flatten()[1::2]
                    )
                    B_acc = (
                        acc_cond[1, 1::2]
                        if acc_cond.ndim == 2
                        else np.asarray(acc_cond[1]).flatten()[1::2]
                    )
                    if len(A1_acc) >= 6 and len(B_acc) >= 6:
                        final_A1 = float(np.mean(A1_acc[-6:]))
                        init_B = float(np.mean(B_acc[:6]))
                        transfer_records.append(
                            {
                                "run_id": run_id,
                                "init_scale": init_scale,
                                "sparsity": sparsity,
                                "sparsity_label": sp_label,
                                "condition": cond_name,
                                "participant": pid,
                                "error_diff": init_B - final_A1,
                            }
                        )
                if acc_cond.size > 0:
                    n_ph = acc_cond.shape[0]
                    for ph in range(min(3, n_ph)):
                        phase_name = ["post_A", "post_B", "post_A2"][ph]
                        mean_acc = float(np.nanmean(acc_cond[ph]))
                        accuracy_records.append(
                            {
                                "run_id": run_id,
                                "init_scale": init_scale,
                                "sparsity_label": sp_label,
                                "condition": cond_name,
                                "phase": phase_name,
                                "participant": pid,
                                "mean_accuracy": mean_acc,
                            }
                        )
                for phase_idx, phase_name in [
                    (0, "post_A"),
                    (1, "post_B"),
                    (2, "post_A2"),
                ]:
                    if f"hiddens_post_phase_{phase_idx}" not in entry:
                        continue
                    for path in ("combined", "core", "comms"):
                        try:
                            hids = ann._get_last_step_geometry_hids(
                                entry, phase_idx, path=path
                            )
                        except (KeyError, ValueError):
                            continue
                        metrics = ann.compute_state_representation_metrics(
                            hids, variance_thresholds=(0.95, 0.99), top_k=2
                        )
                        row = {
                            "run_id": run_id,
                            "init_scale": init_scale,
                            "sparsity_label": sp_label,
                            "condition": cond_name,
                            "participant": pid,
                            "phase": phase_name,
                            "pathway": path,
                            "var_topk": metrics["var_topk"],
                        }
                        for thr, n in metrics["n_components"].items():
                            row[f"n_pcs_{int(thr * 100)}"] = n
                        rep_records.append(row)

    transfer_df = pd.DataFrame(transfer_records)
    accuracy_df = pd.DataFrame(accuracy_records)
    rep_df = pd.DataFrame(rep_records)
    angles_all = pd.concat(angles_records, ignore_index=True) if angles_records else pd.DataFrame()
    return transfer_df, accuracy_df, rep_df, angles_all


def _sem(s: pd.Series) -> float:
    x = s.dropna()
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def _pivot_mean_sem(
    df: pd.DataFrame,
    value: str,
    index: str,
    columns: str,
    subset_col: str | None = None,
    subset_val: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df
    if subset_col is not None:
        d = d[d[subset_col] == subset_val]
    g = d.groupby([index, columns], as_index=False)[value]
    mean = g.mean().pivot(index=index, columns=columns, values=value)
    sem = g.apply(lambda s: _sem(s)).reset_index(name="_sem")
    sem = sem.pivot(index=index, columns=columns, values="_sem")
    return mean, sem


def _fmt_table(mean: pd.DataFrame, sem: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    lines = []
    for idx in mean.index:
        parts = [str(idx)]
        for col in mean.columns:
            m = mean.loc[idx, col]
            s = sem.loc[idx, col] if idx in sem.index and col in sem.columns else np.nan
            if np.isnan(m):
                parts.append("—")
            elif np.isnan(s):
                parts.append(float_fmt.format(m))
            else:
                parts.append(f"{float_fmt.format(m)} ± {float_fmt.format(s)}")
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def run_pipeline(routing: str) -> dict:
    root = _project_root()
    sys.path.insert(0, str(root))
    from a1b2.analysis import transfer_interference as ann

    config_path = root / "a1b2" / "models" / "experiments.json"
    with open(config_path) as f:
        settings = json.load(f)

    rows = _select_rows(settings, routing)
    df = pd.DataFrame(rows)
    df["sparsity_label"] = pd.Categorical(
        df["sparsity_label"], categories=SPARSITY_ORDER, ordered=True
    )
    valid = df[df["path_exists"]].copy()

    all_ann = _load_all_ann_data(valid, ann)
    transfer_df, accuracy_df, rep_df, angles_all = _build_frames(all_ann, ann)

    return {
        "routing": routing,
        "n_selected": len(df),
        "n_valid_paths": len(valid),
        "n_runs_loaded": len(all_ann),
        "transfer_df": transfer_df,
        "accuracy_df": accuracy_df,
        "rep_df": rep_df,
        "angles_all": angles_all,
    }


def main():
    root = _project_root()
    sys.path.insert(0, str(root))

    for routing in ("shared", "task_routed"):
        out = run_pipeline(routing)
        print(f"\n## ROUTING={routing}")
        print(
            f"selected_conditions={out['n_selected']} valid_paths={out['n_valid_paths']} "
            f"runs_with_same_near_far={out['n_runs_loaded']}"
        )
        tdf = out["transfer_df"]
        adf = out["accuracy_df"]
        rdf = out["rep_df"]
        ang = out["angles_all"]
        print(f"transfer rows={len(tdf)} accuracy rows={len(adf)} rep rows={len(rdf)} angle rows={len(ang)}")

        # N per (sparsity, init_scale, condition) for transfer
        if not tdf.empty:
            cnt = (
                tdf.groupby(["sparsity_label", "init_scale", "condition"], as_index=False)
                .size()
                .rename(columns={"size": "n"})
            )
            min_n = cnt["n"].min()
            max_n = cnt["n"].max()
            sparse_cells = cnt[cnt["n"] < 50]
            print(f"transfer cell n: min={min_n} max={max_n} cells_with_n<50: {len(sparse_cells)}")

        # Global mean error_diff by condition
        if not tdf.empty:
            g = tdf.groupby("condition")["error_diff"].agg(["mean", "std", "count"])
            print("\n### Transfer error_diff (global by similarity)")
            print(g.to_string())

        # Mean transfer by sparsity × condition (collapse init)
        if not tdf.empty:
            print("\n### Mean transfer by sparsity_label × condition (all init_scale)")
            pivot = tdf.groupby(["sparsity_label", "condition"])["error_diff"].mean().unstack("condition")
            print(pivot.reindex(SPARSITY_ORDER, level=0).to_string())

        # Mean transfer by init × condition at sp=1.0 and no_comms
        for sp in ("1.0", "no_comms", "single_module"):
            if not tdf.empty and sp in set(tdf["sparsity_label"].astype(str)):
                sub = tdf[tdf["sparsity_label"].astype(str) == sp]
                print(f"\n### Mean transfer init_scale × condition | sparsity={sp}")
                p = sub.groupby(["init_scale", "condition"])["error_diff"].mean().unstack("condition")
                print(p.reindex(INIT_SCALE_ORDER).to_string())

        # post_B accuracy: collapse run to mean per cell
        if not adf.empty:
            b = adf[adf["phase"] == "post_B"]
            print("\n### Mean post_B accuracy by sparsity × condition")
            p = b.groupby(["sparsity_label", "condition"])["mean_accuracy"].mean().unstack("condition")
            print(p.reindex(SPARSITY_ORDER).to_string())

        # Principal angles
        if not ang.empty:
            print("\n### Mean principal angle (deg) by sparsity × condition")
            p = ang.groupby(["sparsity_label", "condition"])["principal_angle_between"].mean().unstack("condition")
            print(p.reindex(SPARSITY_ORDER).to_string())

        # Interaction: at init_scale 1.0 and 0.01, transfer by sp × condition
        for isc in (1.0, 0.01):
            if not tdf.empty:
                sub = tdf[np.isclose(tdf["init_scale"].astype(float), isc)]
                if sub.empty:
                    continue
                print(f"\n### Mean transfer sparsity × condition | init_scale={isc}")
                p = sub.groupby(["sparsity_label", "condition"])["error_diff"].mean().unstack("condition")
                print(p.reindex(SPARSITY_ORDER).to_string())

        # rep: combined, post_B, var_topk
        if not rdf.empty:
            rb = rdf[(rdf["pathway"] == "combined") & (rdf["phase"] == "post_B")]
            print("\n### Mean var_topk (combined post_B) by sparsity × condition")
            p = rb.groupby(["sparsity_label", "condition"])["var_topk"].mean().unstack("condition")
            print(p.reindex(SPARSITY_ORDER).to_string())


if __name__ == "__main__":
    main()
