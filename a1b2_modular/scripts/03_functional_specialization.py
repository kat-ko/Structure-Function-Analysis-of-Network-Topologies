# Compute functional specialization metrics (retraining, correlation, ablations) for a run.
# Usage (from a1b2_modular or project root with --base-folder):
#   python scripts/03_functional_specialization.py <run_id>
#   python scripts/03_functional_specialization.py two_module_rnn_50 --no-ablations
#   python scripts/03_functional_specialization.py two_module_rnn_50 --participants study1_same_sub20

import argparse
import json
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from a1b2.analysis import run_loader
from a1b2.analysis import transfer_interference as ann
from a1b2.analysis.retraining_a1b2 import (
    REGRESSION_CHANCE,
    create_retraining_model_a1b2,
    train_probe_readout_a1b2,
    eval_probe_readout_a1b2,
    retraining_specialization_scalar,
    compute_ablations_metric_a1b2,
)
from a1b2.analysis.correlations_a1b2 import compute_correlation_metric_a1b2
from a1b2.data.basic_funcs import get_datasets
from a1b2.models.ffn import CreateParticipantDataset


def _project_root(base_folder):
    return os.path.abspath(os.path.normpath(base_folder))


def build_loader_for_participant(df, participant, task_parameters, batch_size=32, shuffle=False):
    """Build a single DataLoader over A1 + B + A2 for one participant."""
    dataset_A1, dataset_B, dataset_A2, _, _ = get_datasets(df, participant, task_parameters)
    combined = ConcatDataset([
        CreateParticipantDataset(dataset_A1),
        CreateParticipantDataset(dataset_B),
        CreateParticipantDataset(dataset_A2),
    ])
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(
        description="Compute functional specialization metrics for a simulation run."
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run id (e.g. two_module_rnn_50) or path to sim folder.",
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        default="./",
        help="Base project folder (default: current dir).",
    )
    parser.add_argument(
        "--participants",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these participant ids. Default: all with state+npz.",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip retraining-specialization metric.",
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="Skip correlation-specialization metric.",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip ablation-specialization metric.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for loaders (default: 32).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Epochs for probe readout training (default: 5).",
    )
    parser.add_argument(
        "--regression-chance",
        type=float,
        default=REGRESSION_CHANCE,
        help="Expected accuracy of random predictor for chance correction (default: %.2f)." % REGRESSION_CHANCE,
    )
    args = parser.parse_args()

    root = _project_root(args.base_folder)
    data_folder = os.path.join(root, "data")
    # Allow run_id to be a path to the sim folder
    if os.path.isdir(args.run_id):
        sim_folder = os.path.abspath(args.run_id)
        run_id = os.path.basename(sim_folder.rstrip(os.sep))
    else:
        run_id = args.run_id
        sim_folder = os.path.join(data_folder, "simulations", run_id)

    if not os.path.isdir(sim_folder):
        raise SystemExit(f"Sim folder not found: {sim_folder}")

    settings = run_loader.load_settings(sim_folder)
    task_parameters = settings.get("task_parameters")
    if not task_parameters:
        task_parameters = ann.setup_task_parameters()

    df = ann.load_participant_data(data_folder)
    available_participants = set(df["participant"].unique())
    participants_with_state = run_loader.list_participants_with_state(sim_folder)
    if args.participants is not None:
        participants_with_state = [p for p in participants_with_state if p in args.participants]
    participants = [p for p in participants_with_state if p in available_participants]
    if not participants:
        raise SystemExit(
            "No participants found with both state_*.pt and sim_*.npz and present in trial data."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for participant in participants:
        row = {"id": participant}
        if not args.no_retrain:
            row["retraining_specialization"] = None
        if not args.no_correlation:
            row["correlation_specialization"] = None
        if not args.no_ablations:
            row["ablation_specialization"] = None

        loader = build_loader_for_participant(
            df, participant, task_parameters,
            batch_size=args.batch_size, shuffle=True,
        )
        wrapper = run_loader.build_wrapper_from_settings(settings, device=device)
        state_path = os.path.join(sim_folder, f"state_{participant}.pt")
        run_loader.load_wrapper_state(wrapper, state_path)

        if not args.no_retrain:
            create_retraining_model_a1b2(wrapper, device=device)
            train_probe_readout_a1b2(
                wrapper, loader, settings.get("condition", {}),
                n_epochs=args.n_epochs, lr=1e-3, device=device,
            )
            acc = eval_probe_readout_a1b2(wrapper, loader, device)
            row["retraining_specialization"] = float(retraining_specialization_scalar(
                acc[0, 0], acc[1, 0], acc[0, 1], acc[1, 1], chance=args.regression_chance
            ))

        if not args.no_correlation:
            npz_path = os.path.join(sim_folder, f"sim_{participant}.npz")
            with np.load(npz_path, allow_pickle=True) as data:
                if "hiddens_per_module" not in data or "probes" not in data:
                    row["correlation_specialization"] = None
                else:
                    participant_data = {
                        "hiddens_per_module": data["hiddens_per_module"],
                        "probes": data["probes"],
                        "inputs": data["inputs"],
                    }
                    out = compute_correlation_metric_a1b2(participant_data, n_samples=10)
                    row["correlation_specialization"] = float(out["correlation_specialization"])

        if not args.no_ablations:
            if args.no_retrain:
                # Need probe readout for ablations; build and train if we skipped retrain
                create_retraining_model_a1b2(wrapper, device=device)
                train_probe_readout_a1b2(
                    wrapper, loader, settings.get("condition", {}),
                    n_epochs=args.n_epochs, lr=1e-3, device=device,
                )
            ab = compute_ablations_metric_a1b2(wrapper, loader, device, chance=args.regression_chance)
            row["ablation_specialization"] = float(ab["ablation_specialization"])

        results.append(row)

    # Aggregate
    mean_sem = {}
    for key in ("retraining_specialization", "correlation_specialization", "ablation_specialization"):
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            mean_sem[key] = {"mean": float(np.mean(vals)), "sem": float(np.std(vals) / (len(vals) ** 0.5))}
        else:
            mean_sem[key] = {"mean": None, "sem": None}

    out = {
        "run_id": run_id,
        "participants": results,
        "mean": mean_sem,
    }
    out_path = os.path.join(sim_folder, "specialization_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
