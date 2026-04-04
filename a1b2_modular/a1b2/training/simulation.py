"""
Run simulation for participant learning: A1 -> B -> A2, save .npz per participant.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from a1b2.data.basic_funcs import get_datasets, get_clockwise_order
from a1b2.models.ffn import CreateParticipantDataset, batch_to_torch
from a1b2.training.schedule import runSchedule, train_participant_schedule


def run_simulation(
    training_params,
    network_params,
    task_parameters,
    df,
    do_test,
    dosave=0,
    sim_folder=None,
    arch="ffn",
    condition=None,
):
    """
    Run neural network simulation for participant learning.

    1. For each participant: get datasets, build loaders, run A1 -> B -> A2 via runSchedule.
    2. Optionally save results to sim_<participant>.npz in sim_folder.

    Parameters
    ----------
    training_params : list
        [participants, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr]
    network_params : list
        [dim_input, dim_hidden, dim_output] (dim_hidden for FFN; for two_module_rnn use 2*hidden_size)
    task_parameters : dict
        Task config (e.g. nStim_perTask, schedules).
    df : pandas.DataFrame
        Participant trial data.
    do_test : int
        Whether to run test trials (1) or not (0).
    dosave : int
        1 to save .npz per participant, 0 not to save.
    sim_folder : str or path
        Directory for saving .npz files when dosave=1.
    arch : str
        "ffn", "single_module_rnn", or "two_module_rnn".
    condition : dict or None
        Condition config; required for RNN archs (dim_hidden, n_modules, sparsity, common_input, common_readout, etc.).

    Returns
    -------
    list
        List of participant result dicts.
    """
    dim_input, dim_hidden, dim_output = network_params
    participants, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params

    results = []

    for participant in tqdm(participants, desc="Participants"):
        dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels = get_datasets(df, participant, task_parameters)

        A_inputs = raw_inputs[0]
        B_inputs = raw_inputs[1]
        A_labels_feat1 = raw_labels[0, 0:2].T
        B_labels_feat1 = raw_labels[1, 0:2].T
        ordered_indices_A = get_clockwise_order(A_labels_feat1)
        ordered_indices_B = get_clockwise_order(B_labels_feat1)
        ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

        trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
        trainloader_B = DataLoader(CreateParticipantDataset(dataset_B), batch_size=batch_size, shuffle=shuffle)
        trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)

        network = None
        init_weights_fn = None
        rnn_extra = None
        if arch in ("two_module_rnn", "single_module_rnn") and condition is not None:
            from a1b2.models.two_module_rnn import TwoModuleRNNWrapper
            hidden_size = condition.get("dim_hidden", 50)
            n_modules = condition.get("n_modules", 2 if arch == "two_module_rnn" else 1)
            # Per-layer width for last-layer state (n_modules * hidden_size). Do not multiply by n_layers:
            # runSchedule buffers match PyTorch RNN final hidden shape per timestep.
            dim_hidden_rnn = n_modules * hidden_size
            n_layers = int(condition.get("n_layers", 1))
            dropout = float(condition.get("dropout", 0.0))
            if n_layers < 1:
                raise ValueError(f"condition n_layers must be >= 1, got {n_layers}")
            if not 0.0 <= dropout <= 1.0:
                raise ValueError(f"condition dropout must be in [0, 1], got {dropout}")
            network = TwoModuleRNNWrapper(
                input_size=dim_input,
                output_size=dim_output,
                hidden_size=hidden_size,
                n_modules=n_modules,
                n_layers=n_layers,
                dropout=dropout,
                sparsity=condition.get("sparsity", 1.0),
                common_input=condition.get("common_input", False),
                common_readout=condition.get("common_readout", True),
                cell_type=condition.get("cell_type", "RNN"),
                input_routing=condition.get("input_routing", "shared"),
            )
            init_scale = condition.get("init_scale")
            if init_scale is not None:
                from a1b2.models.rnn_init import apply_init_scale
                init_scope = condition.get("init_scope", condition.get("init_policy", "global"))
                init_weights_fn = lambda net: apply_init_scale(net, init_scale, scope=init_scope)
            rnn_extra = {"n_modules": n_modules, "hidden_size": hidden_size}
            forward_kwargs_from_batch = None
            if condition.get("input_routing") == "task_routed":
                forward_kwargs_from_batch = lambda data: {"feature_probe": batch_to_torch(data["feature_probe"])}
            nb_steps = condition.get("nb_steps", 1)
            return_trajectory = nb_steps > 1
            participant_results = runSchedule(
                train_participant_schedule,
                lr,
                gamma,
                n_epochs,
                dim_input,
                dim_hidden_rnn,
                dim_output,
                trainloader_A1,
                trainloader_B,
                trainloader_A2,
                ordered_inputs,
                do_test,
                network=network,
                init_weights_fn=init_weights_fn,
                rnn_extra=rnn_extra,
                forward_kwargs_from_batch=forward_kwargs_from_batch,
                nb_steps=nb_steps,
                return_trajectory=return_trajectory,
            )
        else:
            participant_results = runSchedule(
                train_participant_schedule,
                lr,
                gamma,
                n_epochs,
                dim_input,
                dim_hidden,
                dim_output,
                trainloader_A1,
                trainloader_B,
                trainloader_A2,
                ordered_inputs,
                do_test,
            )

        participant_results["participant"] = participant

        if dosave and sim_folder is not None:
            file_path = f"{sim_folder}/sim_{participant}.npz"
            save_dict = {k: v for k, v in participant_results.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(file_path, **save_dict)
            if arch in ("two_module_rnn", "single_module_rnn") and network is not None:
                state_path = os.path.join(sim_folder, f"state_{participant}.pt")
                torch.save(network.state_dict(), state_path)

        results.append(participant_results)

    return results
