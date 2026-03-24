"""
Within-phase screening metrics from sim_*.npz (task-routed A↔B proxies).

Used by notebooks for aggregated tables; parse participant IDs into task similarity
(study1_same_sub1 → same; geom_sub_far_2 → far).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def parse_task_similarity_from_participant(participant_id: str) -> Optional[str]:
    """
    Map participant string to schedule similarity: same | near | far.

    Supported patterns (from preprocess batches and geometry helpers):
    - ``study1_far_sub12`` / ``study2_near_sub3`` → middle token is similarity
    - ``geom_sub_same``, ``geom_sub_far_1`` → token after geom_sub_

    Returns None if no match (caller should QC).
    """
    s = str(participant_id).strip().lower()

    # Human-data batches: study{1|2}_{same|near|far}_sub{idx}
    m = re.match(r"^study[12]_(same|near|far)_sub\d+", s)
    if m:
        return m.group(1)

    # Geometry phantom participants from generate_geometry_df
    m = re.match(r"^geom_sub_(same|near|far)(?:_\d+)?$", s)
    if m:
        return m.group(1)

    # Loose fallback: geom_sub_XXX anywhere
    m = re.search(r"geom_sub_(same|near|far)", s)
    if m:
        return m.group(1)

    return None


def parse_study_cohort_from_participant(participant_id: str) -> str:
    """Broad data source: study1, study2, geom, or unknown."""
    s = str(participant_id).strip().lower()
    if re.match(r"^study1_", s):
        return "study1"
    if re.match(r"^study2_", s):
        return "study2"
    if s.startswith("geom_sub_"):
        return "geom"
    return "unknown"


def trial_mask_1d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    return np.isfinite(x)


def load_sim_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def summarize_npz_for_overview(
    path: Path, phase_b: int = 1, phase_a2: int = 2
) -> Dict[str, Any]:
    """
    Return a flat dict of screening metrics for one sim_*.npz file.

    Expects keys: losses, accuracy, probes; optional hiddens_per_module.
    """
    row: Dict[str, Any] = {
        "participant": path.stem.replace("sim_", ""),
        "path": str(path.name),
    }
    try:
        d = load_sim_npz(path)
    except Exception as e:
        row["error"] = str(e)
        return row

    for k in ("losses", "accuracy", "probes"):
        if k not in d:
            row["error"] = f"missing {k}"
            return row

    losses = d["losses"]
    acc = d["accuracy"]
    probes = d["probes"]
    hpm = d.get("hiddens_per_module")
    mod_a, mod_b = 0, 1

    def phase_masks(phase: int):
        m = trial_mask_1d(losses[phase])
        return m

    m_b = phase_masks(phase_b)
    pr_b = probes[phase_b, m_b].astype(int)
    u, cts = np.unique(pr_b, return_counts=True)
    row["b_n_valid"] = int(m_b.sum())
    for uu, cc in zip(u, cts):
        row[f"b_n_probe_{int(uu)}"] = int(cc)

    if hpm is not None and m_b.sum() > 0:
        h = hpm[phase_b, m_b]
        n0 = np.linalg.norm(h[:, mod_a, :], axis=1)
        n1 = np.linalg.norm(h[:, mod_b, :], axis=1)
        row["b_median_hA_over_hB_all"] = float(np.nanmedian(n0 / (n1 + 1e-8)))
    else:
        row["b_median_hA_over_hB_all"] = np.nan

    m_b1 = m_b & (probes[phase_b] == 1)
    if hpm is not None and m_b1.sum() > 0:
        h1 = hpm[phase_b, m_b1]
        n0 = np.linalg.norm(h1[:, mod_a, :], axis=1)
        n1 = np.linalg.norm(h1[:, mod_b, :], axis=1)
        row["b_median_hA_over_hB_probe1"] = float(np.nanmedian(n0 / (n1 + 1e-8)))
        row["b_mean_loss_probe1"] = float(np.nanmean(losses[phase_b, m_b1]))
        row["b_mean_acc_probe1"] = float(np.nanmean(acc[phase_b, m_b1]))
    else:
        row["b_median_hA_over_hB_probe1"] = np.nan
        row["b_mean_loss_probe1"] = np.nan
        row["b_mean_acc_probe1"] = np.nan

    m_a2 = phase_masks(phase_a2)
    for pv in (0, 1):
        sub = m_a2 & (probes[phase_a2] == pv)
        if sub.sum() > 0:
            row[f"a2_mean_loss_probe{pv}"] = float(
                np.nanmean(losses[phase_a2, sub])
            )
            row[f"a2_mean_acc_probe{pv}"] = float(np.nanmean(acc[phase_a2, sub]))
        else:
            row[f"a2_mean_loss_probe{pv}"] = np.nan
            row[f"a2_mean_acc_probe{pv}"] = np.nan

    l0 = row.get("a2_mean_loss_probe0", np.nan)
    l1 = row.get("a2_mean_loss_probe1", np.nan)
    if np.isfinite(l0) and np.isfinite(l1):
        row["a2_loss_gap_p1_minus_p0"] = float(l1 - l0)
    else:
        row["a2_loss_gap_p1_minus_p0"] = np.nan

    row["has_core_comms_keys"] = "hiddens_post_phase_0_core_per_module" in d

    return row


def collect_overview_for_run_folder(run_folder: Path) -> List[Dict[str, Any]]:
    rows = []
    for npz_path in sorted(run_folder.glob("sim_*.npz")):
        row = summarize_npz_for_overview(npz_path)
        row["run_folder"] = run_folder.name
        row["run_folder_path"] = str(run_folder.resolve())
        rows.append(row)
    return rows


def build_overview_long(
    run_folders: List[Path],
) -> pd.DataFrame:
    """
    Scan multiple simulation directories; add similarity and cohort from participant id.
    """
    all_rows: List[Dict[str, Any]] = []
    for folder in run_folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for row in collect_overview_for_run_folder(folder):
            pid = row["participant"]
            row["task_similarity"] = parse_task_similarity_from_participant(pid)
            row["study_cohort"] = parse_study_cohort_from_participant(pid)
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def discover_task_routed_folders(
    simulations_root: Path,
    name_must_contain: str = "task_routed",
    require_npz: bool = True,
) -> List[Path]:
    """List immediate child dirs under simulations_root that look like task-routed runs."""
    simulations_root = Path(simulations_root)
    if not simulations_root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(simulations_root.iterdir()):
        if not p.is_dir():
            continue
        if name_must_contain.lower() not in p.name.lower():
            continue
        if require_npz and not any(p.glob("sim_*.npz")):
            continue
        out.append(p)
    return out
