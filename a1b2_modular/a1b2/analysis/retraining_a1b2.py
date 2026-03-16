"""
Retraining and ablations for A1B2 two-module RNN (regression probe readout).
Bená-style: 3-head probe (only M0, only M1, both), MSE loss, regression accuracy -> specialization scalar.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from a1b2.models.ffn import batch_to_torch

# Expected accuracy of a random predictor for circular regression (1 - min(angular_error/pi, 1)).
# Used for chance correction so the specialization scalar is "above chance" and in [0, 1].
REGRESSION_CHANCE = 0.5


def readout_mask(n_modules, n_in, n_out, ag_to_mask=None):
    """
    Create a mask for the readout layer: zero one module's input slice (ag_to_mask) so only the other module is used.
    Same logic as dynspec: head "only M0" uses ag_to_mask=1 (zero module 1); "only M1" uses ag_to_mask=0.

    Parameters
    ----------
    n_modules : int
        Number of modules.
    n_in : int
        Total readout input size (n_modules * hidden_size).
    n_out : int
        Readout output size (e.g. 4).
    ag_to_mask : int or None
        Module index to zero out. None = use both (no masking).

    Returns
    -------
    torch.Tensor
        Shape (n_out, n_in), 0 where masked, 1 elsewhere.
    """
    # Per-module block size
    block = n_in // n_modules
    mask = torch.ones(n_out, n_in)
    if ag_to_mask is not None:
        mask[:, ag_to_mask * block : (ag_to_mask + 1) * block] = 0
    return mask


class ProbeReadout(nn.Module):
    """Three heads: only M0, only M1, both. Each head Linear(hidden_size*n_modules, output_size) with optional mask."""

    def __init__(self, n_modules, hidden_size, output_size=4):
        super().__init__()
        self.n_modules = n_modules
        self.hidden_size = hidden_size
        self.output_size = output_size
        total_hidden = n_modules * hidden_size
        self.heads = nn.ModuleList([nn.Linear(total_hidden, output_size, bias=False) for _ in range(3)])
        # Head 0: only M0 (zero module 1) -> ag_to_mask=1
        # Head 1: only M1 (zero module 0) -> ag_to_mask=0
        # Head 2: both -> no mask
        masks = [
            readout_mask(n_modules, total_hidden, output_size, ag_to_mask=1),
            readout_mask(n_modules, total_hidden, output_size, ag_to_mask=0),
            readout_mask(n_modules, total_hidden, output_size, ag_to_mask=None),
        ]
        self.register_buffer("masks", torch.stack(masks))

    def forward(self, hidden):
        """
        hidden : (batch, n_modules * hidden_size)
        Returns : list of 3 tensors, each (batch, output_size)
        """
        out = []
        for i in range(3):
            w = self.heads[i].weight * self.masks[i]
            out.append(torch.nn.functional.linear(hidden, w, self.heads[i].bias))
        return out


def create_retraining_model_a1b2(wrapper, device=None):
    """
    Replace the Community's readout with a 3-head probe readout; freeze core and comms.

    Parameters
    ----------
    wrapper : TwoModuleRNNWrapper
        Trained wrapper (state already loaded).
    device : torch.device or None
        If not None, move wrapper to device.

    Returns
    -------
    TwoModuleRNNWrapper
        Same wrapper with community.readout replaced by ProbeReadout; core/comms frozen.
    """
    if device is not None:
        wrapper = wrapper.to(device)
    community = wrapper.community
    n_modules = community.n_modules
    hidden_size = community.hidden_size
    output_size = getattr(community, "output_size", 4)
    if isinstance(output_size, list):
        output_size = output_size[0] if isinstance(output_size[0], int) else 4
    community.readout = ProbeReadout(n_modules, hidden_size, output_size)
    if device is not None:
        community.readout = community.readout.to(device)
    for n, p in community.named_parameters():
        if "readout" not in n:
            p.requires_grad = False
    return wrapper


def _community_hidden(wrapper, x_expanded):
    """Run community core + comms, return last hidden state (batch, n_modules*hidden_size)."""
    with torch.no_grad():
        core_out, comms_out = wrapper.community.core(x_expanded), wrapper.community.comms(x_expanded)
        all_states = core_out[0] + comms_out[0], core_out[1] + comms_out[1]
    # all_states is (h, c) or just h for RNN; h has shape (seq, batch, hidden*n_modules)
    h = all_states[0]
    return h[-1]


def forward_probe(wrapper, x, feature_probe=None):
    """
    Run wrapper's community input expansion, then core+comms, then probe readout.
    x : (batch, 12)
    feature_probe : (batch,) or scalar, for task_routed input expansion.
    Returns : list of 3 tensors, each (batch, 4).
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if wrapper.input_routing == "task_routed" and feature_probe is not None:
        x_expanded = wrapper._routed_input(x, feature_probe)
    else:
        x_expanded = x.repeat(1, 1, wrapper.n_modules)
    hidden = _community_hidden(wrapper, x_expanded)
    return wrapper.community.readout(hidden)


def train_probe_readout_a1b2(wrapper, loader, condition, n_epochs=5, lr=1e-3, device=None):
    """
    Train the 3-head probe readout with MSE on the probed feature only.

    Parameters
    ----------
    wrapper : TwoModuleRNNWrapper
        Wrapper with ProbeReadout already installed (from create_retraining_model_a1b2).
    loader : DataLoader
        Yields dict with "input", "label_x", "label_y", "feature_probe".
    condition : dict
        Unused; kept for API compatibility.
    n_epochs : int
    lr : float
    device : torch.device or None

    Returns
    -------
    wrapper
        Same wrapper with trained probe readout.
    """
    if device is None:
        device = next(wrapper.parameters()).device
    wrapper = wrapper.to(device)
    wrapper.train()
    opt = torch.optim.AdamW(wrapper.community.readout.parameters(), lr=lr)
    for _ in range(n_epochs):
        for batch in loader:
            inp = batch_to_torch(batch["input"]).to(device)
            label_x = batch_to_torch(batch["label_x"]).to(device)
            label_y = batch_to_torch(batch["label_y"]).to(device)
            probe = batch_to_torch(batch["feature_probe"]).to(device)
            if probe.dim() == 0:
                probe = probe.expand(inp.shape[0])
            opt.zero_grad()
            outs = forward_probe(wrapper, inp, probe)
            # Target: (batch, 4) with [cos_f0, sin_f0, cos_f1, sin_f1]; we only have the probed feature's label
            target = torch.stack([label_x, label_y], dim=1)
            loss = torch.tensor(0.0, device=device)
            for head_out in outs:
                # For each sample, MSE on the probed feature slice only
                for b in range(inp.shape[0]):
                    f = int(probe[b].item())
                    pred_slice = head_out[b, f * 2 : (f + 1) * 2]
                    loss = loss + torch.nn.functional.mse_loss(pred_slice, target[b])
            loss = loss / (3 * inp.shape[0])
            loss.backward()
            opt.step()
    return wrapper


def regression_accuracy_a1b2(pred_4d, label_x, label_y, feature_probe):
    """
    Per-sample accuracy as 1 - min(angular_error/pi, 1). Averages over batch.

    pred_4d : (batch, 4) [cos_f0, sin_f0, cos_f1, sin_f1]
    label_x, label_y : (batch,) cos and sin of the probed feature's angle.
    feature_probe : (batch,) 0 or 1.
    """
    device = pred_4d.device
    if not isinstance(label_x, torch.Tensor):
        label_x = torch.tensor(label_x, dtype=torch.float32, device=device)
    if not isinstance(label_y, torch.Tensor):
        label_y = torch.tensor(label_y, dtype=torch.float32, device=device)
    if not isinstance(feature_probe, torch.Tensor):
        feature_probe = torch.tensor(feature_probe, device=device)
    if label_x.dim() == 0:
        label_x = label_x.unsqueeze(0)
    if label_y.dim() == 0:
        label_y = label_y.unsqueeze(0)
    if feature_probe.dim() == 0:
        feature_probe = feature_probe.expand(pred_4d.shape[0])
    batch = pred_4d.shape[0]
    errs = []
    for b in range(batch):
        f = int(feature_probe[b].item())
        px, py = pred_4d[b, f * 2], pred_4d[b, f * 2 + 1]
        angle_pred = math.atan2(py.item(), px.item())
        angle_true = math.atan2(label_y[b].item(), label_x[b].item())
        diff = abs(math.atan2(math.sin(angle_pred - angle_true), math.cos(angle_pred - angle_true)))
        errs.append(min(diff / math.pi, 1.0))
    return 1.0 - (sum(errs) / batch)


def eval_probe_readout_a1b2(wrapper, loader, device=None):
    """
    Eval mode; for each head and each feature (0, 1), compute mean regression accuracy
    over trials where feature_probe matches that feature.
    Returns acc[probe_idx, feature_idx] with shape (3, 2).
    """
    if device is None:
        device = next(wrapper.parameters()).device
    wrapper.eval()
    # Accumulate: per (probe, feature) list of accs
    accs = [[[] for _ in range(2)] for _ in range(3)]
    with torch.no_grad():
        for batch in loader:
            inp = batch_to_torch(batch["input"]).to(device)
            label_x = batch_to_torch(batch["label_x"]).to(device)
            label_y = batch_to_torch(batch["label_y"]).to(device)
            probe = batch_to_torch(batch["feature_probe"]).to(device)
            if probe.dim() == 0:
                probe = probe.expand(inp.shape[0])
            outs = forward_probe(wrapper, inp, probe)
            for h, head_out in enumerate(outs):
                for f in range(2):
                    mask = probe == f
                    if mask.any():
                        pred_sub = head_out[mask]
                        lx = label_x[mask] if label_x.dim() > 0 else label_x.expand_as(probe)[mask]
                        ly = label_y[mask] if label_y.dim() > 0 else label_y.expand_as(probe)[mask]
                        pr_sub = probe[mask]
                        acc = regression_accuracy_a1b2(pred_sub, lx, ly, pr_sub)
                        accs[h][f].append(acc)
    out = np.zeros((3, 2))
    for h in range(3):
        for f in range(2):
            if accs[h][f]:
                out[h, f] = np.mean(accs[h][f])
            else:
                out[h, f] = np.nan
    return out


def metric_norm_acc(acc, chance=0.0):
    """
    Chance-correct and clip for stability.
    For chance < 1: returns (acc - chance) / (1 - chance) clipped to [1e-5, 1], or 1e-5 if acc <= chance.
    For chance >= 1: no correction, returns clip(acc, 1e-5, 1).
    """
    acc = float(acc)
    if chance >= 1.0:
        return float(np.clip(acc, 1e-5, 1.0))
    if acc <= chance:
        return 1e-5
    denom = 1.0 - chance
    m = (acc - chance) / denom
    return float(np.clip(m, 1e-5, 1.0))


def diff_metric(pair):
    """(a - b) / (a + b)."""
    a, b = float(pair[0]), float(pair[1])
    s = a + b
    if s <= 0:
        return 0.0
    return (a - b) / s


def global_diff_metric(metric_task0, metric_task1):
    """|diff(metric_task0) - diff(metric_task1)| / 2."""
    return abs(diff_metric(metric_task0) - diff_metric(metric_task1)) / 2


def retraining_specialization_scalar(acc_M0_f0, acc_M1_f0, acc_M0_f1, acc_M1_f1, chance=None):
    """
    From probe 0 and 1 accs per feature, compute Bená-style specialization scalar.
    acc_M0_f0 = accuracy when using only module 0 for feature 0, etc.
    chance : expected accuracy of random predictor; default REGRESSION_CHANCE (0.5) for circular regression.
    """
    if chance is None:
        chance = REGRESSION_CHANCE
    m0_f0 = metric_norm_acc(acc_M0_f0, chance)
    m1_f0 = metric_norm_acc(acc_M1_f0, chance)
    m0_f1 = metric_norm_acc(acc_M0_f1, chance)
    m1_f1 = metric_norm_acc(acc_M1_f1, chance)
    return global_diff_metric((m0_f0, m1_f0), (m0_f1, m1_f1))


def compute_ablations_metric_a1b2(wrapper_with_probe_readout, loader, device=None, chance=None):
    """
    Evaluate 3 heads (only M0, only M1, both) on loader; same regression acc as retraining.
    Returns acc[probe_idx, feature_idx] shape (3, 2) and the specialization scalar.
    chance : passed to retraining_specialization_scalar; default REGRESSION_CHANCE.
    """
    acc = eval_probe_readout_a1b2(wrapper_with_probe_readout, loader, device)
    # Probes 0 and 1 are only M0 and only M1
    scalar = retraining_specialization_scalar(
        acc[0, 0], acc[1, 0], acc[0, 1], acc[1, 1], chance=chance
    )
    return {"acc": acc, "ablation_specialization": scalar}


def ablation_specialization_scalar(acc_M0_f0, acc_M1_f0, acc_M0_f1, acc_M1_f1, chance=None):
    """Same formula as retraining_specialization_scalar."""
    return retraining_specialization_scalar(acc_M0_f0, acc_M1_f0, acc_M0_f1, acc_M1_f1, chance=chance)
