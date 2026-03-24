"""
Helpers for hypothesis-aligned views of schema-v2 during-training arrays.

Assumes standard A1-B-A2 phase indices: 0=A1, 1=B, 2=A2. For task_routed setups,
module 0 ↔ task A, module 1 ↔ task B (see TwoModuleRNNWrapper._routed_input).

These functions operate on arrays already loaded from ``sim_*.npz``; they do not
run the model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def trial_mask_finite_1d(a: np.ndarray) -> np.ndarray:
    """Boolean mask of finite entries along the last axis (e.g. valid trials)."""
    x = np.asarray(a, dtype=float)
    return np.isfinite(x)


def comms_m0_over_m1_l2(during_comms_l2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Ratio of comms L2 norms: module0 / (module1 + eps).

    Parameters
    ----------
    during_comms_l2 : array, shape (n_phase, n_trials, n_modules)
    """
    c = np.asarray(during_comms_l2, dtype=np.float32)
    if c.shape[-1] < 2:
        raise ValueError("during_comms_l2 must have n_modules >= 2")
    return (c[..., 0] / (c[..., 1] + np.float32(eps))).astype(np.float32)


def slice_phase_valid(
    losses: np.ndarray,
    probes: np.ndarray,
    phase: int,
    probe_value: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Return boolean mask and indices for valid trials in a phase, optionally filtered by probe.
    """
    m = trial_mask_finite_1d(losses[phase])
    if probe_value is not None:
        m = m & (probes[phase] == probe_value)
    return m


def phase_B_probe1_comms_ratio(
    losses: np.ndarray,
    probes: np.ndarray,
    during_comms_l2: np.ndarray,
    phase_b: int = 1,
) -> Dict[str, Any]:
    """
    **B-phase, feature_probe==1 trials**: comms energy ratio m0/m1.

    Interpretation (task_routed): while training B (input to module 1), relative comms
    magnitude on module 0 vs module 1 can proxy **A-side / cross-module** influence on
    the comms pathway during B learning.
    """
    m = slice_phase_valid(losses, probes, phase_b, probe_value=1)
    ratio = comms_m0_over_m1_l2(during_comms_l2)
    return {
        "phase": phase_b,
        "mask": m,
        "comms_m0_over_m1": ratio[phase_b, m],
        "n_trials": int(m.sum()),
    }


def phase_A2_by_probe(
    losses: np.ndarray,
    probes: np.ndarray,
    during_comms_l2: np.ndarray,
    phase_a2: int = 2,
) -> Dict[str, Any]:
    """
    **A2 phase**: split comms ratio by feature_probe (0 vs 1).

    Probe 0 trials: A-feature training signal; probe 1: B-feature **without** gradient
    (when do_update==2). Comparing comms patterns across probes supports **B→A** style
    hypotheses during A2.
    """
    ratio = comms_m0_over_m1_l2(during_comms_l2)
    out: Dict[str, Any] = {"phase": phase_a2}
    for pv in (0, 1):
        m = slice_phase_valid(losses, probes, phase_a2, probe_value=pv)
        out[f"probe_{pv}_n"] = int(m.sum())
        out[f"probe_{pv}_comms_m0_over_m1"] = ratio[phase_a2, m]
    return out


def summarize_during_npz(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flat QC / summary dict for one loaded npz mapping (keys from np.load).

    Safe if during_* keys are missing.
    """
    row: Dict[str, Any] = {}
    if "during_comms_l2" not in d:
        row["has_during_training"] = False
        return row
    row["has_during_training"] = True
    row["during_comms_l2_shape"] = str(np.asarray(d["during_comms_l2"]).shape)
    if "during_comms_m0_over_m1_l2" in d:
        row["has_during_comms_ratio"] = True
    else:
        row["has_during_comms_ratio"] = False
    if "losses" in d and "probes" in d:
        losses = np.asarray(d["losses"])
        probes = np.asarray(d["probes"])
        dcl = np.asarray(d["during_comms_l2"])
        try:
            b = phase_B_probe1_comms_ratio(losses, probes, dcl)
            x = b["comms_m0_over_m1"]
            if x.size:
                row["b_probe1_median_comms_m0_m1"] = float(np.nanmedian(x))
        except Exception:
            row["b_probe1_median_comms_m0_m1"] = np.nan
    return row
