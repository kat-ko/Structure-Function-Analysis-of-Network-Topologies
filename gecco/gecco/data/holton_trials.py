"""
Holton et al. trial tables → torch-friendly tensors.

Vendored from transfer-interference / a1b2 lineage (src/utils/basic_funcs.get_datasets
and helpers) so gecco does not depend on a1b2_modular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple


def setup_task_parameters() -> Dict[str, Any]:
    return {"nStim_perTask": 6, "schedules": ["same", "near", "far"]}


def load_trial_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "task_section" in df.columns:
        df.loc[df["task_section"] == "B", "test_trial"] = 0
        df = df.loc[df["task_section"].isin(["A1", "B", "A2"])].copy()
    return df


def pick_participants(
    df: pd.DataFrame,
    *,
    conditions: Sequence[str],
    per_condition: int = 2,
    seed: int = 0,
) -> List[str]:
    """Deterministic small subset: up to `per_condition` pids per `condition` value."""
    rng = np.random.default_rng(seed)
    out: List[str] = []
    for cond in conditions:
        sub = df.loc[df["condition"] == cond, "participant"].drop_duplicates()
        ids = sub.tolist()
        rng.shuffle(ids)
        out.extend(ids[:per_condition])
    return sorted(set(out))


def filter_participant_data(df: pd.DataFrame, participant: str, task_section: str) -> pd.DataFrame:
    return df.loc[
        (df["participant"] == participant) & (df["task_section"] == task_section),
        ["index", "feature_idx", "feat_val", "noisy_feedback_value", "stimID", "test_trial"],
    ].reset_index(drop=True)


def adjust_indices(participant_data: pd.DataFrame, offset: int) -> pd.DataFrame:
    participant_data = participant_data.copy()
    participant_data["index"] -= offset
    return participant_data.reset_index(drop=True)


def create_inputs_matrix(participant_data: pd.DataFrame, n_stim_per_task: int) -> np.ndarray:
    length = participant_data.shape[0]
    inputs = np.zeros((length, n_stim_per_task * 2), dtype=np.float32)
    for index, row in participant_data.iterrows():
        inputs[index, int(row["stimID"])] = 1.0
    return inputs


def process_raw_inputs_and_labels(
    participant_data: pd.DataFrame, n_stim_per_task: int, task_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    unique_inputs = participant_data["stimID"].unique().astype(int)
    raw_inputs = np.full((n_stim_per_task, n_stim_per_task * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((4, n_stim_per_task), np.nan, dtype=np.float32)

    for idx, stim_id in enumerate(unique_inputs):
        feat1 = participant_data.loc[
            (participant_data["stimID"] == stim_id) & (participant_data["feature_idx"] == 0), "feat_val"
        ].unique()
        feat2 = participant_data.loc[
            (participant_data["stimID"] == stim_id) & (participant_data["feature_idx"] == 1), "feat_val"
        ].unique()
        raw_labels[0, idx] = np.cos(feat1)[0]
        raw_labels[1, idx] = np.sin(feat1)[0]
        raw_labels[2, idx] = np.cos(feat2)[0]
        raw_labels[3, idx] = np.sin(feat2)[0]

        input_skeleton = np.zeros((n_stim_per_task * 2), dtype=np.float32)
        input_skeleton[stim_id] = 1.0
        raw_inputs[idx, :] = input_skeleton

    return raw_inputs, raw_labels


def assemble_dataset(
    participant_data: pd.DataFrame,
    inputs: np.ndarray,
    label_cos: np.ndarray,
    label_sin: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "index": participant_data["index"].values.astype(np.int64),
        "stim_index": participant_data["stimID"].values.astype(np.int64),
        "input": inputs.astype(np.float32),
        "feature_probe": participant_data["feature_idx"].values.astype(np.int64),
        "test_stim": participant_data["test_trial"].values.astype(np.int64),
        "label_x": label_cos.astype(np.float32),
        "label_y": label_sin.astype(np.float32),
    }


def get_datasets(
    df: pd.DataFrame, participant: str, task_parameters: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    n = task_parameters["nStim_perTask"]
    pA1 = filter_participant_data(df, participant, "A1")
    pB = filter_participant_data(df, participant, "B")
    pA2 = filter_participant_data(df, participant, "A2")
    a_len = len(pA1)
    pB = adjust_indices(pB, a_len)
    pA2 = adjust_indices(pA2, a_len + len(pB))

    A1_inputs = create_inputs_matrix(pA1, n)
    B_inputs = create_inputs_matrix(pB, n)
    A2_inputs = create_inputs_matrix(pA2, n)

    raw_inputs = np.full((3, n, n * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((3, 4, n), np.nan, dtype=np.float32)
    raw_inputs[0], raw_labels[0] = process_raw_inputs_and_labels(pA1, n, 0)
    raw_inputs[1], raw_labels[1] = process_raw_inputs_and_labels(pB, n, 1)
    raw_inputs[2], raw_labels[2] = process_raw_inputs_and_labels(pA2, n, 2)

    dataset_A1 = assemble_dataset(
        pA1, A1_inputs, np.cos(pA1["feat_val"].values), np.sin(pA1["feat_val"].values)
    )
    dataset_B = assemble_dataset(pB, B_inputs, np.cos(pB["feat_val"].values), np.sin(pB["feat_val"].values))
    dataset_A2 = assemble_dataset(
        pA2, A2_inputs, np.cos(pA2["feat_val"].values), np.sin(pA2["feat_val"].values)
    )
    return dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels


def subsample_dataset_dict(ds: Dict[str, np.ndarray], max_trials: Optional[int], seed: int) -> Dict[str, np.ndarray]:
    n = ds["input"].shape[0]
    if max_trials is None or max_trials >= n:
        return {k: v.copy() for k, v in ds.items()}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_trials, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in ds.items()}


def dataset_dict_to_arrays(ds: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {
        "input": torch.from_numpy(ds["input"]),
        "feature_probe": torch.from_numpy(ds["feature_probe"]),
        "label_x": torch.from_numpy(ds["label_x"]),
        "label_y": torch.from_numpy(ds["label_y"]),
    }


def build_cyclic_ab_schedule(
    dataset_A1: Dict[str, np.ndarray],
    dataset_B: Dict[str, np.ndarray],
    *,
    pattern: str = "ABABAB",
    max_trials_per_segment: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Repeat A1 and B segments in alternation (default A-B-A-B-A-B).

    LBA simplification: every ``A`` block reuses the **A1** trial tensor (not A2 retest-only
    updates). Full Holton A2 asymmetry can be reintroduced later by inserting A2 slices.
    """
    A = subsample_dataset_dict(dataset_A1, max_trials_per_segment, seed)
    B = subsample_dataset_dict(dataset_B, max_trials_per_segment, seed + 1)
    segs: List[Dict[str, np.ndarray]] = []
    for ch in pattern:
        if ch == "A":
            segs.append(A)
        elif ch == "B":
            segs.append(B)
        else:
            raise ValueError(f"Unknown pattern char {ch!r}")
    merged: Dict[str, List[np.ndarray]] = {k: [] for k in A.keys()}
    for s in segs:
        for k in merged:
            merged[k].append(s[k])
    stacked = {k: np.concatenate(merged[k], axis=0) for k in merged}
    return dataset_dict_to_arrays(stacked)
