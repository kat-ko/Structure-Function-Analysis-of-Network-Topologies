"""
Feedforward network for A1-B-A2 task (simpleLinearNet).
"""
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

from a1b2.data.basic_funcs import wrap_to_pi


class simpleLinearNet(nn.Module):
    """One hidden layer, no bias. Forward returns (out, hid)."""
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(simpleLinearNet, self).__init__()
        self.in_hid = nn.Linear(dim_input, dim_hidden, bias=False)
        self.hid_out = nn.Linear(dim_hidden, dim_output, bias=False)

    def forward(self, x):
        hid = self.in_hid(x)
        out = self.hid_out(hid)
        return out, hid


def ex_initializer_(model, gamma=1e-3, mean=0.0):
    """In-place init: hidden layer std=gamma, output layer std=1e-3."""
    for name, param in model.named_parameters():
        if "weight" in name:
            if "hid_out" in name:
                std = 1e-3
            else:
                std = gamma
            nn.init.normal_(param, mean=mean, std=std)


class CreateParticipantDataset(Dataset):
    """PyTorch Dataset for participant data."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {key: self.dataset[key][idx] for key in self.dataset}
        if self.transform:
            sample = self.transform(sample)
        return sample


def compute_accuracy(predictions, ground_truth):
    """Accuracy in radians (1 - normalized error)."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    wrapped_difference = wrap_to_pi(predictions - ground_truth)
    normalized_error = np.abs(wrapped_difference) / np.pi
    return 1 - normalized_error


def batch_to_torch(numpy_version):
    """Convert numpy batch to torch float tensor."""
    return numpy_version.type(torch.FloatTensor)


def ordered_sweep(network, ranked_inputs, nb_steps=1, return_trajectory=False, return_core_comms=False):
    """Run network on ordered inputs; return (preds, hids) numpy, optionally (preds, hids, trajectory) or (preds, hids, trajectory, core_hid, comms_hid) when return_core_comms."""
    try:
        device = next(network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if isinstance(ranked_inputs, np.ndarray):
        x = torch.from_numpy(ranked_inputs).float().to(device)
    else:
        x = ranked_inputs.float().to(device) if ranked_inputs.device != device else ranked_inputs
    if nb_steps > 1:
        from a1b2.data.temporal import temporal_data
        x_temporal, _ = temporal_data(x, nb_steps=nb_steps, noise_ratio=None)
        result = network(x_temporal, return_trajectory=return_trajectory, return_core_comms=return_core_comms)
    else:
        if return_core_comms:
            result = network(x, return_core_comms=True)
        else:
            result = network(x)
    preds = result[0].detach().cpu().numpy().copy()
    hids = result[1].detach().cpu().numpy().copy()
    if len(result) == 5:
        traj = result[2].detach().cpu().numpy().copy() if result[2] is not None else None
        core_hid = result[3].detach().cpu().numpy().copy()
        comms_hid = result[4].detach().cpu().numpy().copy()
        return preds, hids, traj, core_hid, comms_hid
    if len(result) == 3:
        return preds, hids, result[2].detach().cpu().numpy().copy()
    return preds, hids
