# Run simulations for a condition from experiments.json.
# Usage (from a1b2_modular or project root with --base-folder):
#   python scripts/02_run_simulations.py rich_50
#   python scripts/02_run_simulations.py two_module_rnn_50
# Geometry (single participant):
#   python scripts/02_run_simulations.py rich_50 --geometry
#   python scripts/02_run_simulations.py rich_50 --geometry --participant study1_same_sub20

import os
import sys
from pathlib import Path

# Ensure a1b2 is importable when run from a1b2_modular (scripts/ is on path, not parent)
_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import argparse
import numpy as np

from a1b2.data.basic_funcs import set_seed
from a1b2.training.simulation import run_simulation
from a1b2.training.schedule import train_single_schedule
from a1b2.analysis import transfer_interference as ann
from a1b2.utils.run_config import build_run_id
from a1b2.utils.sim_storage import resolve_sim_root


def _project_root(base_folder):
    return os.path.abspath(os.path.normpath(base_folder))


def load_settings(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def run_experiment(condition_name, base_folder="./", storage_mode="auto", print_output_path=False):
    set_seed(2024)
    root = _project_root(base_folder)
    data_folder = os.path.join(root, "data")
    config_path = os.path.join(root, "a1b2", "models", "experiments.json")

    settings = load_settings(config_path)
    condition = next((c for c in settings["conditions"] if c["name"] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")

    df = ann.load_participant_data(data_folder)
    participants = df["participant"].unique()
    task_parameters = ann.setup_task_parameters()

    dim_input = task_parameters["nStim_perTask"] * 2
    arch = condition.get("arch", "ffn")

    # Default for recurrent architectures: run with nb_steps = 2 unless explicitly overridden.
    if arch in ("two_module_rnn", "single_module_rnn") and "nb_steps" not in condition:
        condition["nb_steps"] = 2
    if arch in ("two_module_rnn", "single_module_rnn"):
        n_modules = condition.get("n_modules", 2 if arch == "two_module_rnn" else 1)
        dim_hidden = n_modules * condition.get("dim_hidden", 50)
    else:
        dim_hidden = condition["dim_hidden"]
    dim_output = 4
    network_params = [dim_input, dim_hidden, dim_output]

    training_params = [
        list(participants),
        settings["n_phase"],
        settings["n_epochs"],
        settings["n_epochs"] * (task_parameters["nStim_perTask"] * 2) * 10,
        settings["shuffle"],
        settings["batch_size"],
        condition.get("gamma", 0.01),
        settings["learning_rate"],
    ]

    run_id = build_run_id(condition)
    sim_root = resolve_sim_root(condition, Path(data_folder), mode=storage_mode)
    sim_folder = str(sim_root / run_id)

    if print_output_path:
        print(f"Condition: {condition_name}")
        print(f"run_id: {run_id}")
        print(f"storage_mode: {storage_mode}")
        print(f"output_path: {sim_folder}")
        return

    os.makedirs(sim_folder, exist_ok=True)

    settings_to_save = {
        "condition": condition,
        "training_params": {
            "participants": [str(p) for p in participants],
            "n_phase": settings["n_phase"],
            "n_epochs": settings["n_epochs"],
            "n_train_trials": settings["n_epochs"] * (task_parameters["nStim_perTask"] * 2) * 10,
            "shuffle": settings["shuffle"],
            "batch_size": settings["batch_size"],
            "gamma": condition.get("gamma", 0.01),
            "lr": settings["learning_rate"],
        },
        "network_params": network_params,
        "task_parameters": task_parameters,
    }
    settings_to_save = ann.numpy_to_python(settings_to_save)
    with open(os.path.join(sim_folder, "settings.json"), "w") as f:
        json.dump(settings_to_save, f, indent=4)

    print(f"Starting simulation for condition: {condition_name} (arch={arch})")
    print(f"Number of participants: {len(participants)}")
    print(f"Network: input={dim_input}, hidden={dim_hidden}, output={dim_output}")

    run_simulation(
        training_params,
        network_params,
        task_parameters,
        df,
        do_test=1,
        dosave=1,
        sim_folder=sim_folder,
        arch=arch,
        condition=condition if arch in ("two_module_rnn", "single_module_rnn") else None,
    )


def run_geometry_experiment(condition_name, participant_to_copy="study1_same_sub20", base_folder="./"):
    root = _project_root(base_folder)
    data_folder = os.path.join(root, "data")
    config_path = os.path.join(root, "a1b2", "models", "experiments.json")

    settings = load_settings(config_path)
    condition = next((c for c in settings["conditions"] if c["name"] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")

    df = ann.load_participant_data(data_folder)
    task_parameters = ann.setup_task_parameters()
    dim_input = task_parameters["nStim_perTask"] * 2
    dim_hidden = condition["dim_hidden"]
    dim_output = 4
    network_params = [dim_input, dim_hidden, dim_output]

    training_params = [
        [participant_to_copy],
        settings["n_phase"],
        settings["n_epochs"],
        settings["n_epochs"] * (task_parameters["nStim_perTask"] * 2) * 10,
        settings["shuffle"],
        settings["batch_size"],
        condition.get("gamma", 0.01),
        settings["learning_rate"],
    ]

    geometry_df = ann.generate_geometry_df(
        df, participant_to_copy, near_rule=np.pi / 6, far_rule=np.pi
    )

    print(f"Starting geometry simulation for condition: {condition_name}")
    print(f"Using participant schedule: {participant_to_copy}")

    geom_results = train_single_schedule(
        training_params, network_params, task_parameters, geometry_df, do_test=0
    )

    output_path = os.path.join(data_folder, "simulations", f"geom_results_{condition_name}.npz")
    np.savez_compressed(output_path, **geom_results)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run neural network simulations")
    parser.add_argument("condition", type=str, help="Condition name (e.g. rich_50, two_module_rnn_50)")
    parser.add_argument("--base-folder", type=str, default="./", help="Base project folder")
    parser.add_argument(
        "--storage-mode",
        type=str,
        default="auto",
        choices=["auto", "primary", "ablation"],
        help="Where to store outputs: auto routes by primary-grid policy.",
    )
    parser.add_argument(
        "--print-output-path",
        action="store_true",
        help="Print resolved output folder and exit (no training).",
    )
    parser.add_argument("--geometry", action="store_true", help="Run geometry visualization experiment")
    parser.add_argument("--participant", type=str, default="study1_same_sub20", help="Participant ID for geometry")
    args = parser.parse_args()

    if args.geometry:
        run_geometry_experiment(args.condition, args.participant, args.base_folder)
    else:
        run_experiment(args.condition, args.base_folder, args.storage_mode, args.print_output_path)


if __name__ == "__main__":
    main()
