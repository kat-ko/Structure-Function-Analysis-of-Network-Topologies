"""
Regression: log_during_training must not change training tensors (losses, hiddens, predictions).
Smoke: when True, during_* keys exist and align with hiddens_per_module shape.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from a1b2.models.ffn import CreateParticipantDataset
from a1b2.models.two_module_rnn import TwoModuleRNNWrapper
from a1b2.training.schedule import train_participant_schedule


def _tiny_b_dataset(n_trials=6, dim_input=12, feature_probe=1):
    t = np.linspace(0, 2 * np.pi, n_trials, dtype=np.float32)
    return {
        "index": np.arange(n_trials, dtype=np.int64),
        "stim_index": np.arange(n_trials, dtype=np.int64),
        "input": np.random.RandomState(0).randn(n_trials, dim_input).astype(np.float32),
        "label_x": np.cos(t),
        "label_y": np.sin(t),
        "feature_probe": np.full(n_trials, feature_probe, dtype=np.int64),
        "test_stim": np.zeros(n_trials, dtype=np.int64),
    }


def test_log_during_training_regression_matches_primary_tensors():
    torch.manual_seed(0)
    np.random.seed(0)
    dim_input = 12
    hidden_size = 8
    n_modules = 2
    rnn_extra = {"n_modules": n_modules, "hidden_size": hidden_size}

    net = TwoModuleRNNWrapper(
        input_size=dim_input,
        output_size=4,
        hidden_size=hidden_size,
        n_modules=n_modules,
        sparsity=1.0,
        common_input=False,
        common_readout=True,
        cell_type="RNN",
        input_routing="task_routed",
    )
    net.train()
    loader = DataLoader(
        CreateParticipantDataset(_tiny_b_dataset()),
        batch_size=2,
        shuffle=False,
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    fwd = lambda data: {"feature_probe": torch.as_tensor(data["feature_probe"], dtype=torch.long)}

    def run(log_flag):
        net2 = TwoModuleRNNWrapper(
            input_size=dim_input,
            output_size=4,
            hidden_size=hidden_size,
            n_modules=n_modules,
            sparsity=1.0,
            common_input=False,
            common_readout=True,
            cell_type="RNN",
            input_routing="task_routed",
        )
        net2.load_state_dict(net.state_dict())
        opt2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        return train_participant_schedule(
            net2,
            loader,
            n_epochs=1,
            loss_function=loss_fn,
            optimizer=opt2,
            do_update=1,
            do_test=0,
            forward_kwargs_from_batch=fwd,
            nb_steps=1,
            device=torch.device("cpu"),
            rnn_extra=rnn_extra,
            log_during_training=log_flag,
            during_log_post_step=True,
        )

    out_off = run(False)
    out_on = run(True)
    idxs = [5, 6, 7, 8]
    names = ["losses", "accuracy", "predictions", "hiddens"]
    for i, name in zip(idxs, names):
        a, b = out_off[i], out_on[i]
        np.testing.assert_allclose(
            np.asarray(a),
            np.asarray(b),
            rtol=1e-5,
            atol=1e-6,
            err_msg=name,
        )


def test_during_log_pre_step_matches_log_off_tensors():
    """Single forward + detach (during_log_post_step=False) must not change loss/hidden tensors vs log off."""
    torch.manual_seed(2)
    np.random.seed(2)
    dim_input = 12
    hidden_size = 8
    n_modules = 2
    rnn_extra = {"n_modules": n_modules, "hidden_size": hidden_size}
    base = TwoModuleRNNWrapper(
        input_size=dim_input,
        output_size=4,
        hidden_size=hidden_size,
        n_modules=n_modules,
        sparsity=1.0,
        common_input=False,
        common_readout=True,
        cell_type="RNN",
        input_routing="task_routed",
    )
    torch.manual_seed(3)
    base_state = base.state_dict()
    loader = DataLoader(
        CreateParticipantDataset(_tiny_b_dataset()),
        batch_size=2,
        shuffle=False,
    )
    loss_fn = nn.MSELoss()
    fwd = lambda data: {"feature_probe": torch.as_tensor(data["feature_probe"], dtype=torch.long)}

    def run_variant(log: bool, post_step: bool):
        net2 = TwoModuleRNNWrapper(
            input_size=dim_input,
            output_size=4,
            hidden_size=hidden_size,
            n_modules=n_modules,
            sparsity=1.0,
            common_input=False,
            common_readout=True,
            cell_type="RNN",
            input_routing="task_routed",
        )
        net2.load_state_dict(base_state)
        opt2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        return train_participant_schedule(
            net2,
            loader,
            n_epochs=1,
            loss_function=loss_fn,
            optimizer=opt2,
            do_update=1,
            do_test=0,
            forward_kwargs_from_batch=fwd,
            nb_steps=1,
            device=torch.device("cpu"),
            rnn_extra=rnn_extra,
            log_during_training=log,
            during_log_post_step=post_step,
        )

    out_off = run_variant(False, True)
    out_pre = run_variant(True, False)
    for i in (5, 6, 7, 8):
        np.testing.assert_allclose(
            np.asarray(out_off[i]),
            np.asarray(out_pre[i]),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"index {i}",
        )


def test_forward_return_core_comms_unchanged_output():
    torch.manual_seed(0)
    net = TwoModuleRNNWrapper(
        input_size=12,
        output_size=4,
        hidden_size=8,
        n_modules=2,
        sparsity=1.0,
        common_input=False,
        common_readout=True,
        input_routing="shared",
    )
    net.eval()
    x = torch.randn(1, 3, 12)
    r0 = net(x)
    r1 = net(x, return_core_comms=True)
    np.testing.assert_allclose(r0[0].detach().numpy(), r1[0].detach().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r0[1].detach().numpy(), r1[1].detach().numpy(), rtol=1e-6, atol=1e-6)


def test_during_training_analysis_comms_ratio():
    from a1b2.analysis.during_training import comms_m0_over_m1_l2, phase_B_probe1_comms_ratio

    # (n_phase=2, n_trials=2, n_modules=2)
    dcl = np.array(
        [
            [[1.0, 2.0], [3.0, 1.0]],
            [[2.0, 4.0], [1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    losses = np.zeros((2, 2), dtype=np.float32)
    probes = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    r = comms_m0_over_m1_l2(dcl)
    assert abs(r[0, 0] - 0.5) < 1e-5
    b = phase_B_probe1_comms_ratio(losses, probes, dcl, phase_b=0)
    assert b["n_trials"] == 1
    assert abs(b["comms_m0_over_m1"][0] - 3.0) < 1e-5


def test_log_during_training_populates_during_arrays():
    torch.manual_seed(1)
    dim_input = 12
    hidden_size = 6
    n_modules = 2
    rnn_extra = {"n_modules": n_modules, "hidden_size": hidden_size}
    net = TwoModuleRNNWrapper(
        input_size=dim_input,
        output_size=4,
        hidden_size=hidden_size,
        n_modules=n_modules,
        sparsity=1.0,
        common_input=False,
        common_readout=True,
        cell_type="RNN",
        input_routing="shared",
    )
    net.train()
    loader = DataLoader(
        CreateParticipantDataset(_tiny_b_dataset(n_trials=4, feature_probe=1)),
        batch_size=2,
        shuffle=False,
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    out = train_participant_schedule(
        net,
        loader,
        n_epochs=1,
        loss_function=nn.MSELoss(),
        optimizer=opt,
        do_update=1,
        do_test=0,
        nb_steps=1,
        device=torch.device("cpu"),
        rnn_extra=rnn_extra,
        log_during_training=True,
    )
    dc, dco, dcl2, dcol2 = out[11:15]
    assert dc is not None and dco is not None
    assert dc.shape == (4, n_modules, hidden_size)
    assert out[8].shape == (4, n_modules * hidden_size)  # flat concat hiddens
    assert dcl2.shape == (4, n_modules)
    assert np.isfinite(dc).all()


def test_build_run_id_suffix_for_schema2():
    from a1b2.utils.run_config import build_run_id

    c = {
        "name": "two_module_rnn_50_task_routed",
        "arch": "two_module_rnn",
        "dim_hidden": 50,
        "sparsity": 1.0,
        "common_input": False,
        "common_readout": True,
        "input_routing": "task_routed",
        "nb_steps": 2,
        "log_during_training": True,
    }
    rid = build_run_id(c)
    assert "__s2" in rid
    c2 = dict(c)
    c2["run_id_suffix"] = "ablation"
    assert "ablation" in build_run_id(c2)
