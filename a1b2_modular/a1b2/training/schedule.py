"""
A1-B-A2 schedule: runSchedule, train_participant_schedule, train_single_schedule.
Uses FFN (simpleLinearNet) by default; architecture-agnostic where possible.
"""
import math
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from a1b2.data.basic_funcs import get_datasets, get_clockwise_order
from a1b2.models.ffn import (
    simpleLinearNet,
    ex_initializer_,
    CreateParticipantDataset,
    compute_accuracy,
    batch_to_torch,
    ordered_sweep,
)


def _split_hiddens_by_module(hiddens, n_modules, hidden_size):
    """Split (..., n_modules*hidden_size) into (..., n_modules, hidden_size)."""
    shape = hiddens.shape
    out = hiddens.reshape(shape[:-1] + (n_modules, hidden_size))
    return out


def train_participant_schedule(
    network, trainloader, n_epochs, loss_function, optimizer, do_update, do_test,
    forward_kwargs_from_batch=None,
    nb_steps=1,
    device=None,
):
    """
    Train the network on one phase (A1, B, or A2).
    Expects network to have forward(x) -> (out, hid). Optionally in_hid/hid_out for embeddings/readouts.
    When forward_kwargs_from_batch is provided, call network(input_t, **forward_kwargs_from_batch(data)).
    When nb_steps > 1, build temporal input (nb_steps, batch, input_size) before each forward.
    """
    if device is None:
        device = next(network.parameters()).device

    metrics = {
        "indexes": [],
        "losses": [],
        "accuracy": [],
        "predictions": [],
        "hiddens": [],
        "embeddings": [],
        "readouts": [],
        "probes": [],
        "test_stim": [],
        "labels": [],
        "inputs": [],
    }

    for epoch in range(n_epochs):
        for batch_idx, data in enumerate(trainloader):
            optimizer.zero_grad()

            index = data['stim_index']
            input_t = batch_to_torch(data['input']).to(device)
            label_x = batch_to_torch(data['label_x']).to(device)
            label_y = batch_to_torch(data['label_y']).to(device)
            feature_probe = batch_to_torch(data['feature_probe']).to(device)
            test_stim = batch_to_torch(data['test_stim']).to(device)

            joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)
            radians_label = np.arctan2(label_x.cpu().numpy(), label_y.cpu().numpy())

            forward_kwargs = forward_kwargs_from_batch(data) if forward_kwargs_from_batch else {}
            forward_kwargs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in forward_kwargs.items()}
            if nb_steps > 1:
                from a1b2.data.temporal import temporal_data
                x_temporal, _ = temporal_data(input_t, nb_steps=nb_steps, noise_ratio=None)
                result = network(x_temporal, **forward_kwargs)
            else:
                result = network(input_t, **forward_kwargs)
            out, hid = result[0], result[1]

            probe_val = int(feature_probe.reshape(-1)[0].cpu().numpy())
            if probe_val == 0:
                loss = loss_function(out[:, :2], joined_label)
                pred_rads = np.arctan2(out[:, 0].detach().cpu().numpy(), out[:, 1].detach().cpu().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
            elif probe_val == 1:
                loss = loss_function(out[:, 2:4], joined_label)
                pred_rads = np.arctan2(out[:, 2].detach().cpu().numpy(), out[:, 3].detach().cpu().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
            else:
                raise ValueError("Undefined loss setting for feature_probe.")

            test_stim_np = test_stim.cpu().numpy()
            if do_update == 1 and do_test == 1 and np.all(test_stim_np == 0):
                loss.backward()
                optimizer.step()
            elif do_update == 1 and do_test == 0:
                loss.backward()
                optimizer.step()
            elif do_update == 2 and probe_val == 0:
                loss.backward()
                optimizer.step()

            batch_size = input_t.shape[0]
            metrics["indexes"].append(np.atleast_1d(np.asarray(index)))
            metrics["inputs"].append(input_t.cpu().numpy())
            metrics["labels"].append(joined_label.cpu().numpy())
            metrics["probes"].append(np.broadcast_to(feature_probe.cpu().numpy(), (batch_size,)))
            metrics["test_stim"].append(np.broadcast_to(test_stim_np, (batch_size,)))
            metrics["losses"].append(np.full(batch_size, loss.item()))
            metrics["accuracy"].append(np.atleast_1d(np.asarray(accuracy)))
            metrics["predictions"].append(out.detach().cpu().numpy())
            metrics["hiddens"].append(hid.detach().cpu().numpy())

            if hasattr(network, 'in_hid') and hasattr(network.in_hid, 'weight'):
                emb = network.in_hid.weight.detach().cpu().numpy()
            else:
                emb = np.zeros((hid.shape[-1], input_t.shape[-1]), dtype=np.float32)
            if hasattr(network, 'hid_out') and hasattr(network.hid_out, 'weight'):
                rd = network.hid_out.weight.detach().cpu().numpy()
            else:
                rd = np.zeros((out.shape[-1], hid.shape[-1]), dtype=np.float32)
            metrics["embeddings"].append(np.tile(emb[np.newaxis, :, :], (batch_size, 1, 1)))
            metrics["readouts"].append(np.tile(rd[np.newaxis, :, :], (batch_size, 1, 1)))

    # Flatten to per-trial arrays (support any batch size)
    n_trials = sum(x.shape[0] for x in metrics["inputs"])
    metrics["inputs"] = np.concatenate(metrics["inputs"], axis=0)
    metrics["labels"] = np.concatenate(metrics["labels"], axis=0)
    metrics["predictions"] = np.concatenate(metrics["predictions"], axis=0)
    metrics["hiddens"] = np.concatenate(metrics["hiddens"], axis=0)
    metrics["indexes"] = np.concatenate(metrics["indexes"], axis=0)
    metrics["probes"] = np.concatenate(metrics["probes"], axis=0)
    metrics["test_stim"] = np.concatenate(metrics["test_stim"], axis=0)
    metrics["losses"] = np.concatenate(metrics["losses"], axis=0)
    metrics["accuracy"] = np.concatenate(metrics["accuracy"], axis=0)
    # Embeddings/readouts: per-trial (repeated per batch)
    metrics["embeddings"] = np.concatenate(metrics["embeddings"], axis=0)
    metrics["readouts"] = np.concatenate(metrics["readouts"], axis=0)

    return (
        metrics["indexes"],
        metrics["inputs"],
        metrics["labels"],
        metrics["probes"],
        metrics["test_stim"],
        metrics["losses"],
        metrics["accuracy"],
        metrics["predictions"],
        metrics["hiddens"],
        metrics["embeddings"],
        metrics["readouts"],
    )


def runSchedule(
    train_function,
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
    network=None,
    init_weights_fn=None,
    rnn_extra=None,
    forward_kwargs_from_batch=None,
    nb_steps=1,
    return_trajectory=False,
):
    """
    Runs A1 -> B -> A2. If network is None, builds simpleLinearNet and initializes with ex_initializer_.
    Otherwise uses provided network and optional init_weights_fn(network).
    When rnn_extra is set (dict with n_modules, hidden_size), per-module hiddens are stored.
    When forward_kwargs_from_batch is set, it is passed to train_function for each batch (e.g. feature_probe for task routing).
    When nb_steps > 1, temporal input is used; when return_trajectory True and RNN, trajectory is stored.
    """
    n_train_trials = n_epochs * dim_input * 10
    n_phase = 3
    n_stim = ordered_inputs.shape[0]

    results = {
        "indexes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "inputs": np.full((n_phase, n_train_trials, dim_input), np.nan, dtype=np.float32),
        "labels": np.full((n_phase, n_train_trials, 2), np.nan, dtype=np.float32),
        "test_stim": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "probes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "losses": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "accuracy": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "predictions": np.full((n_phase, n_train_trials, dim_output), np.nan, dtype=np.float32),
        "hiddens": np.full((n_phase, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
        "embeddings": np.full((n_phase, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
        "readouts": np.full((n_phase, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
    }
    if rnn_extra is not None:
        n_mod, h_size = rnn_extra["n_modules"], rnn_extra["hidden_size"]
        results["hiddens_per_module"] = np.full((n_phase, n_train_trials, n_mod, h_size), np.nan, dtype=np.float32)

    if network is None:
        network = simpleLinearNet(dim_input, dim_hidden, dim_output)
        ex_initializer_(network, gamma)
    elif init_weights_fn is not None:
        init_weights_fn(network)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    sweep_result = ordered_sweep(
        network, ordered_inputs,
        nb_steps=nb_steps,
        return_trajectory=return_trajectory and nb_steps > 1 and rnn_extra is not None,
        return_core_comms=rnn_extra is not None,
    )
    if len(sweep_result) == 5:
        initial_preds, initial_hiddens, initial_trajectory, _, _ = sweep_result
        if initial_trajectory is not None:
            results["hiddens_pre_training_trajectory"] = initial_trajectory
    elif len(sweep_result) == 3:
        initial_preds, initial_hiddens, initial_trajectory = sweep_result
        results["hiddens_pre_training_trajectory"] = initial_trajectory
    else:
        initial_preds, initial_hiddens = sweep_result
    results["preds_pre_training"] = initial_preds
    results["hiddens_pre_training"] = initial_hiddens
    if rnn_extra is not None:
        n_modules = rnn_extra["n_modules"]
        hidden_size = rnn_extra["hidden_size"]
        results["hiddens_pre_training_per_module"] = _split_hiddens_by_module(
            initial_hiddens, n_modules, hidden_size
        )

    phases = [
        (0, trainloader_A1, 1),
        (1, trainloader_B, 1),
        (2, trainloader_A2, 2),
    ]
    for phase, loader, do_update in phases:
        out = train_function(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test,
            forward_kwargs_from_batch=forward_kwargs_from_batch,
            nb_steps=nb_steps,
            device=device,
        )
        (idx, inp, lbl, prb, test, loss, acc, pred, hid, emb, rd) = out
        n = min(len(idx), n_train_trials)
        results["indexes"][phase, :n] = idx[:n]
        results["inputs"][phase, :n, :] = inp[:n]
        results["labels"][phase, :n, :] = lbl[:n]
        results["probes"][phase, :n] = prb[:n]
        results["test_stim"][phase, :n] = test[:n]
        results["losses"][phase, :n] = loss[:n]
        results["accuracy"][phase, :n] = acc[:n]
        results["predictions"][phase, :n, :] = pred[:n]
        results["hiddens"][phase, :n, :] = hid[:n]
        results["embeddings"][phase, :n, :, :] = emb[:n]
        results["readouts"][phase, :n, :, :] = rd[:n]
        if rnn_extra is not None:
            results["hiddens_per_module"][phase, :n, :, :] = _split_hiddens_by_module(
                hid[:n], rnn_extra["n_modules"], rnn_extra["hidden_size"]
            )

        sweep_result = ordered_sweep(
            network, ordered_inputs,
            nb_steps=nb_steps,
            return_trajectory=return_trajectory and nb_steps > 1 and rnn_extra is not None,
            return_core_comms=rnn_extra is not None,
        )
        if len(sweep_result) == 5:
            post_preds, post_hiddens, post_trajectory, core_hid, comms_hid = sweep_result
            if post_trajectory is not None:
                results[f"hiddens_post_phase_{phase}_trajectory"] = post_trajectory
            results[f"hiddens_post_phase_{phase}_core_per_module"] = _split_hiddens_by_module(
                core_hid, rnn_extra["n_modules"], rnn_extra["hidden_size"]
            )
            results[f"hiddens_post_phase_{phase}_comms_per_module"] = _split_hiddens_by_module(
                comms_hid, rnn_extra["n_modules"], rnn_extra["hidden_size"]
            )
        elif len(sweep_result) == 3:
            post_preds, post_hiddens, post_trajectory = sweep_result
            results[f"hiddens_post_phase_{phase}_trajectory"] = post_trajectory
        else:
            post_preds, post_hiddens = sweep_result
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens
        if rnn_extra is not None:
            results[f"hiddens_post_phase_{phase}_per_module"] = _split_hiddens_by_module(
                post_hiddens, rnn_extra["n_modules"], rnn_extra["hidden_size"]
            )

    return results


def train_single_schedule(training_params, network_params, task_parameters, df, do_test):
    """
    Geometry visualization: one A1, then three branches (same/near/far B) then A2.
    Returns results with shape (3, n_phase, ...) for the three B conditions.
    """
    dim_input, dim_hidden, dim_output = network_params
    _, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params

    dataset_A1, dataset_B_same, dataset_A2, raw_inputs, raw_labels = get_datasets(df, "geom_sub_same", task_parameters)
    _, dataset_B_near, _, _, _ = get_datasets(df, "geom_sub_near", task_parameters)
    _, dataset_B_far, _, _, _ = get_datasets(df, "geom_sub_far", task_parameters)

    A_inputs = raw_inputs[0]
    B_inputs = raw_inputs[1]
    A_labels_feat1 = raw_labels[0, 0:2].T
    B_labels_feat1 = raw_labels[1, 0:2].T
    ordered_indices_A = get_clockwise_order(A_labels_feat1)
    ordered_indices_B = get_clockwise_order(B_labels_feat1)
    ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

    trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_same = DataLoader(CreateParticipantDataset(dataset_B_same), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_near = DataLoader(CreateParticipantDataset(dataset_B_near), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_far = DataLoader(CreateParticipantDataset(dataset_B_far), batch_size=batch_size, shuffle=shuffle)
    trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)

    n_train_trials = n_epochs * dim_input * 10
    n_phase_val = 3
    n_stim = task_parameters["nStim_perTask"] * 2

    results = {
        "indexes": np.full((3, n_phase_val, n_train_trials), np.nan, dtype=np.float32),
        "inputs": np.full((3, n_phase_val, n_train_trials, dim_input), np.nan, dtype=np.float32),
        "labels": np.full((3, n_phase_val, n_train_trials, 2), np.nan, dtype=np.float32),
        "test_stim": np.full((3, n_phase_val, n_train_trials), np.nan, dtype=np.float32),
        "probes": np.full((3, n_phase_val, n_train_trials), np.nan, dtype=np.float32),
        "losses": np.full((3, n_phase_val, n_train_trials), np.nan, dtype=np.float32),
        "accuracy": np.full((3, n_phase_val, n_train_trials), np.nan, dtype=np.float32),
        "predictions": np.full((3, n_phase_val, n_train_trials, dim_output), np.nan, dtype=np.float32),
        "hiddens": np.full((3, n_phase_val, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
        "embeddings": np.full((3, n_phase_val, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
        "readouts": np.full((3, n_phase_val, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
        "preds_pre_training": np.full((3, n_stim, dim_output), np.nan, dtype=np.float32),
        "hiddens_pre_training": np.full((3, n_stim, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_0": np.full((3, n_stim, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_0": np.full((3, n_stim, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_1": np.full((3, n_stim, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_1": np.full((3, n_stim, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_2": np.full((3, n_stim, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_2": np.full((3, n_stim, dim_hidden), np.nan, dtype=np.float32),
    }

    network = simpleLinearNet(dim_input, dim_hidden, dim_output)
    ex_initializer_(network, gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    initial_preds, initial_hiddens = ordered_sweep(network, ordered_inputs)
    results["preds_pre_training"] = np.broadcast_to(initial_preds[np.newaxis, :, :], (3, n_stim, dim_output))
    results["hiddens_pre_training"] = np.broadcast_to(initial_hiddens[np.newaxis, :, :], (3, n_stim, dim_hidden))

    out = train_participant_schedule(network, trainloader_A1, n_epochs, loss_function, optimizer, 1, do_test, device=device)
    idx, inp, lbl, prb, test, loss, acc, pred, hid, emb, rd = out
    n = min(len(idx), n_train_trials)
    results["indexes"][0, 0, :n] = idx[:n]
    results["inputs"][0, 0, :n, :] = inp[:n]
    results["labels"][0, 0, :n, :] = lbl[:n]
    results["probes"][0, 0, :n] = prb[:n]
    results["test_stim"][0, 0, :n] = test[:n]
    results["losses"][0, 0, :n] = loss[:n]
    results["accuracy"][0, 0, :n] = acc[:n]
    results["predictions"][0, 0, :n, :] = pred[:n]
    results["hiddens"][0, 0, :n, :] = hid[:n]
    results["embeddings"][0, 0, :n, :, :] = emb[:n]
    results["readouts"][0, 0, :n, :, :] = rd[:n]

    post_preds, post_hiddens = ordered_sweep(network, ordered_inputs)
    results["preds_post_phase_0"][0, :, :] = post_preds
    results["hiddens_post_phase_0"][0, :, :] = post_hiddens

    network_same = copy.deepcopy(network).to(device)
    network_near = copy.deepcopy(network).to(device)
    network_far = copy.deepcopy(network).to(device)
    optimizer_same = torch.optim.SGD(network_same.parameters(), lr=lr)
    optimizer_near = torch.optim.SGD(network_near.parameters(), lr=lr)
    optimizer_far = torch.optim.SGD(network_far.parameters(), lr=lr)

    for condition_idx, (condition_network, condition_loader, condition_optimizer) in enumerate(zip(
        [network_same, network_near, network_far],
        [trainloader_B_same, trainloader_B_near, trainloader_B_far],
        [optimizer_same, optimizer_near, optimizer_far],
    )):
        for phase, loader, do_update in [(1, condition_loader, 1), (2, trainloader_A2, 2)]:
            out = train_participant_schedule(condition_network, loader, n_epochs, loss_function, condition_optimizer, do_update, do_test, device=device)
            idx, inp, lbl, prb, test, loss, acc, pred, hid, emb, rd = out
            n = min(len(idx), n_train_trials)
            results["indexes"][condition_idx, phase, :n] = idx[:n]
            results["inputs"][condition_idx, phase, :n, :] = inp[:n]
            results["labels"][condition_idx, phase, :n, :] = lbl[:n]
            results["probes"][condition_idx, phase, :n] = prb[:n]
            results["test_stim"][condition_idx, phase, :n] = test[:n]
            results["losses"][condition_idx, phase, :n] = loss[:n]
            results["accuracy"][condition_idx, phase, :n] = acc[:n]
            results["predictions"][condition_idx, phase, :n, :] = pred[:n]
            results["hiddens"][condition_idx, phase, :n, :] = hid[:n]
            results["embeddings"][condition_idx, phase, :n, :, :] = emb[:n]
            results["readouts"][condition_idx, phase, :n, :, :] = rd[:n]

            post_preds, post_hiddens = ordered_sweep(condition_network, ordered_inputs)
            results[f"preds_post_phase_{phase}"][condition_idx, :, :] = post_preds
            results[f"hiddens_post_phase_{phase}"][condition_idx, :, :] = post_hiddens

    return results
