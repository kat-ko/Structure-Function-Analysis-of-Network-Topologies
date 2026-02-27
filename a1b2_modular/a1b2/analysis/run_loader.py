"""
Load run settings and trained two-module RNN from simulation folder.
Used by the functional specialization script to rebuild models and list participants.
"""
import json
import os
from pathlib import Path

import torch

from a1b2.models.two_module_rnn import TwoModuleRNNWrapper


def load_settings(sim_folder):
    """
    Read settings.json from a simulation folder.

    Parameters
    ----------
    sim_folder : str or pathlib.Path
        Path to the run folder (e.g. data/simulations/<run_id>).

    Returns
    -------
    dict
        Parsed settings (condition, training_params, network_params, task_parameters).
    """
    sim_folder = Path(sim_folder)
    path = sim_folder / "settings.json"
    with open(path, "r") as f:
        return json.load(f)


def build_wrapper_from_settings(settings, device=None):
    """
    Build TwoModuleRNNWrapper from settings (same args as in simulation.py).

    Parameters
    ----------
    settings : dict
        Must contain "condition" and "network_params" (or "task_parameters" for input size).
    device : torch.device or str or None
        Device to place the model on. If None, model stays on CPU.

    Returns
    -------
    TwoModuleRNNWrapper
        Wrapper on the given device.
    """
    condition = settings["condition"]
    network_params = settings["network_params"]
    dim_input, dim_hidden, dim_output = network_params
    n_modules = condition.get("n_modules", 2)
    hidden_size = condition.get("dim_hidden", 50)

    wrapper = TwoModuleRNNWrapper(
        input_size=dim_input,
        output_size=dim_output,
        hidden_size=hidden_size,
        n_modules=n_modules,
        sparsity=condition.get("sparsity", 1.0),
        common_input=condition.get("common_input", False),
        common_readout=condition.get("common_readout", True),
        cell_type=condition.get("cell_type", "RNN"),
        input_routing=condition.get("input_routing", "shared"),
    )
    if device is not None:
        wrapper = wrapper.to(device)
    return wrapper


def load_wrapper_state(wrapper, state_path):
    """
    Load state_dict from file into the wrapper.

    Parameters
    ----------
    wrapper : TwoModuleRNNWrapper
        Model to load into.
    state_path : str or pathlib.Path
        Path to state_<participant>.pt.

    Returns
    -------
    None
    """
    try:
        device = next(wrapper.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    state = torch.load(state_path, map_location=device)
    wrapper.load_state_dict(state, strict=True)


def list_participants_with_state(sim_folder):
    """
    List participant ids for which both sim_<participant>.npz and state_<participant>.pt exist.

    Parameters
    ----------
    sim_folder : str or pathlib.Path
        Path to the run folder.

    Returns
    -------
    list of str
        Participant ids that have both files.
    """
    sim_folder = Path(sim_folder)
    npz_prefix = "sim_"
    npz_suffix = ".npz"
    state_prefix = "state_"
    state_suffix = ".pt"

    npz_participants = set()
    for name in os.listdir(sim_folder):
        if name.startswith(npz_prefix) and name.endswith(npz_suffix):
            participant = name[len(npz_prefix) : -len(npz_suffix)]
            npz_participants.add(participant)

    state_participants = set()
    for name in os.listdir(sim_folder):
        if name.startswith(state_prefix) and name.endswith(state_suffix):
            participant = name[len(state_prefix) : -len(state_suffix)]
            state_participants.add(participant)

    return sorted(npz_participants & state_participants)


def list_participants_with_npz(sim_folder):
    """
    List participant ids for which sim_<participant>.npz exists.
    Use when state_*.pt are not available (e.g. runs from before state_dict saving).
    """
    sim_folder = Path(sim_folder)
    npz_prefix = "sim_"
    npz_suffix = ".npz"
    out = set()
    for name in os.listdir(sim_folder):
        if name.startswith(npz_prefix) and name.endswith(npz_suffix):
            participant = name[len(npz_prefix) : -len(npz_suffix)]
            out.add(participant)
    return sorted(out)
