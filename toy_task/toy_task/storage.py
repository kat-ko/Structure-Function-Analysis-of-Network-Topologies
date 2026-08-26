"""Per-run artifact storage (Part IV Section 17; Part V Section 7).

Each run writes ``config.json`` (full configuration + behavioral metrics + learning
curves) and ``extractions.npz`` (stacked representation arrays) under
``<out>/<run_id>/``. No reliance on implicit defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .environment import Environment
from .training import RunResult


def run_dir(out_root: str | Path, run_id: str) -> Path:
    p = Path(out_root) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_result(result: RunResult, out_root: str | Path, env: Environment | None = None) -> Path:
    """Persist a :class:`RunResult` to ``<out_root>/<run_id>/``."""
    run_id = result.config["run_id"]
    d = run_dir(out_root, run_id)

    meta = {
        "config": result.config,
        "behavioral": result.behavioral,
        "learning_curves": result.learning_curves,
    }
    if env is not None:
        meta["environment"] = {
            "seed": env.seed,
            "Z": env.Z.tolist(),
            "Z_b": env.Z_b.tolist(),
            "W": env.W.tolist(),
            "b": env.b.tolist(),
            "sigma_train": env.sigma_train,
            "stimulus_regime": env.stimulus_regime,
            "angle_mode": env.angle_mode,
        }
    with open(d / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    _save_extractions(result.extractions, d / "extractions.npz")
    return d


def _save_extractions(extractions: List[dict], path: Path) -> None:
    if not extractions:
        np.savez_compressed(path)
        return
    arrays = {
        "hidden": np.stack([e["hidden"] for e in extractions]),
        "preds": np.stack([e["preds"] for e in extractions]),
        "targets": np.stack([e["targets"] for e in extractions]),
        "phase": np.array([e["phase"] for e in extractions]),
        "task_id": np.array([e["task_id"] for e in extractions]),
        "epoch": np.array([e["epoch"] for e in extractions]),
        "global_epoch": np.array([e["global_epoch"] for e in extractions]),
        "clean_loss_phase": np.array([e["clean_loss_phase"] for e in extractions]),
        "clean_loss_T0": np.array([e["clean_loss_T0"] for e in extractions]),
        "clean_loss_Ts": np.array([e["clean_loss_Ts"] for e in extractions]),
    }
    np.savez_compressed(path, **arrays)


def load_result(out_root: str | Path, run_id: str) -> dict:
    """Load a saved run as a dict of metadata + extraction arrays."""
    d = Path(out_root) / run_id
    with open(d / "config.json") as f:
        meta = json.load(f)
    npz = np.load(d / "extractions.npz", allow_pickle=False)
    meta["extractions"] = {k: npz[k] for k in npz.files}
    return meta
