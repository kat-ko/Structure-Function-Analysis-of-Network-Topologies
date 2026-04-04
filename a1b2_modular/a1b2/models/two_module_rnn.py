"""
Two-module RNN wrapper for A1-B-A2: input_size=12, output_size=4, compatible with runSchedule/simulation.
"""
import torch
import torch.nn as nn

from a1b2.models.community import Community


def build_community_rnn_config(
    input_size=12,
    output_size=4,
    hidden_size=50,
    n_modules=2,
    n_layers=1,
    dropout=0.0,
    cell_type="RNN",
    sparsity=1.0,
    common_input=False,
    common_readout=True,
    binary_comms=False,
):
    """Build config dict for Community (1 or 2 modules) for A1-B-A2."""
    return {
        "input": {"input_size": input_size, "common_input": common_input},
        "modules": {
            "n_modules": n_modules,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "dropout": dropout,
            "cell_type": cell_type,
        },
        "connections": {"sparsity": sparsity, "binary": binary_comms},
        "readout": {
            "output_size": output_size,
            "common_readout": common_readout,
        },
    }


def build_two_module_rnn_config(
    input_size=12,
    output_size=4,
    hidden_size=50,
    n_layers=1,
    dropout=0.0,
    cell_type="RNN",
    sparsity=1.0,
    common_input=False,
    common_readout=True,
    binary_comms=False,
):
    """Build config dict for Community (two modules) for A1-B-A2. Kept for backward compatibility."""
    return build_community_rnn_config(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        n_modules=2,
        n_layers=n_layers,
        dropout=dropout,
        cell_type=cell_type,
        sparsity=sparsity,
        common_input=common_input,
        common_readout=common_readout,
        binary_comms=binary_comms,
    )


class TwoModuleRNNWrapper(nn.Module):
    """
    Wraps Community for A1-B-A2: forward(x) returns (out, hid) with out (batch, 4), hid (batch, n_modules*hidden_size).
    Supports n_modules=1 (single-module / no-module ablation) or n_modules=2.
    Input x: (batch, input_size) or (seq_len, batch, input_size).
    """

    def __init__(
        self,
        input_size=12,
        output_size=4,
        hidden_size=50,
        n_modules=2,
        n_layers=1,
        dropout=0.0,
        cell_type="RNN",
        sparsity=1.0,
        common_input=False,
        common_readout=True,
        binary_comms=False,
        input_routing="shared",
        seed=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_modules = n_modules
        self.hidden_size = hidden_size
        self.common_input = common_input
        self.common_readout = common_readout
        self.input_routing = input_routing

        config = build_community_rnn_config(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            n_modules=n_modules,
            n_layers=n_layers,
            dropout=dropout,
            cell_type=cell_type,
            sparsity=sparsity,
            common_input=common_input,
            common_readout=common_readout,
            binary_comms=binary_comms,
        )
        self.community = Community(config)

        if seed is not None:
            torch.manual_seed(seed)

    def forward(self, x, feature_probe=None, return_trajectory=False, return_core_comms=False):
        # x: (batch, input_size) or (seq_len, batch, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, batch, input_size)
        # Community expects (seq_len, batch, input_size * n_modules)
        if self.input_routing == "task_routed" and feature_probe is not None:
            x_expanded = self._routed_input(x, feature_probe)
        else:
            x_expanded = x.repeat(1, 1, self.n_modules)
        if return_core_comms:
            outputs, all_states, core_final, comms_final = self.community(x_expanded, return_core_comms=True)
        else:
            outputs, all_states = self.community(x_expanded)
            core_final = comms_final = None
        # When common_readout=True: outputs is (seq_len, batch, 4) or (seq_len, batch, 4*n_modules). When common_readout=False: outputs is list of (seq_len, batch, 4) per module.
        if isinstance(outputs, list):
            # Per-module tensors (seq, batch, output_size); sum over modules then take last step -> (batch, output_size)
            out = torch.stack(outputs, dim=0).sum(dim=0)  # (seq_len, batch, output_size)
            out = out[-1]  # (batch, output_size)
        else:
            out = outputs[-1]  # (batch, 4) or (batch, 8)
        # When common_readout=True and single tensor: last dim can be 4*n_modules; collapse to output_size
        if out.dim() == 2 and out.shape[-1] > self.output_size:
            out = out[:, : self.output_size] + out[:, self.output_size :]  # sum modules -> (batch, 4)
        elif out.dim() == 3:
            # (batch, n_modules, output_size) from common_readout=False, or (seq, batch, output_size) from full sequence
            if out.shape[1] == self.n_modules:
                out = out.sum(dim=1)  # (batch, output_size)
            else:
                out = out[-1]  # (seq, batch, output_size) -> (batch, output_size)
        if isinstance(all_states, tuple):
            # final_states: (num_layers, batch, hidden_size * n_modules) — log last layer only for 2D PCA/analysis
            hid = all_states[-1][-1] if all_states[-1].dim() > 1 else all_states[-1]  # (batch, hidden_size * n_modules)
            seq = all_states[0]
        else:
            hid = all_states[-1]
            seq = all_states
        if return_core_comms:
            traj = seq if (return_trajectory and seq.shape[0] > 1) else None
            return out, hid, traj, core_final, comms_final
        if return_trajectory and seq.shape[0] > 1:
            return out, hid, all_states
        return out, hid

    def _routed_input(self, x, feature_probe):
        """Build (seq, batch, input_size*n_modules) with input only to one module per trial."""
        # feature_probe: 0 -> task A -> module 0 only; 1 -> task B -> module 1 only
        device = x.device
        seq_len, batch_size, input_size = x.shape
        out = torch.zeros(seq_len, batch_size, input_size * self.n_modules, device=device, dtype=x.dtype)
        probe = feature_probe if isinstance(feature_probe, torch.Tensor) else torch.tensor(feature_probe, device=device)
        if probe.dim() == 0:
            probe = probe.expand(batch_size)
        for m in range(self.n_modules):
            mask = (probe == m).float().view(1, -1, 1)
            out[:, :, m * input_size : (m + 1) * input_size] = x * mask
        return out
