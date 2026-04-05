"""
Two-module task-routed **pathway** model vs single-module baseline (Holton 12-d one-hot, 4-d output).

Each “module” is a `Linear → tanh` encoder on a **masked** input (fast LBA default; swap in `RNNCell` later).

Routing genome: one bit per input dimension — exclusive assignment to module 0 or 1.
Indices 0–5: task A stimulus slots; 6–11: task B stimulus slots (standard one-hot layout).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def routing_separation_score(genome_bits: List[int]) -> float:
    """Mean routing to module 1 on A slots vs B slots; return |Δ| in [0, 1]."""
    g = torch.tensor(genome_bits, dtype=torch.float32)
    if len(g) != 12:
        raise ValueError("Genome must have length 12 for Holton one-hot layout.")
    a = g[:6].mean().item()
    b = g[6:].mean().item()
    return abs(a - b)


class DualRouteRNN(nn.Module):
    """Two parallel encoders with hard input masks from genome (see module docstring)."""

    def __init__(self, genome_bits: List[int], hidden_per_module: int = 32):
        super().__init__()
        if len(genome_bits) != 12:
            raise ValueError("Expected 12 routing genes.")
        g = torch.tensor(genome_bits, dtype=torch.float32)
        self.register_buffer("mask0", (1.0 - g).unsqueeze(0))  # (1, 12)
        self.register_buffer("mask1", g.unsqueeze(0))

        self.fc0 = nn.Linear(12, hidden_per_module, bias=True)
        self.fc1 = nn.Linear(12, hidden_per_module, bias=True)
        self.readout = nn.Linear(2 * hidden_per_module, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 12)
        h0 = torch.tanh(self.fc0(x * self.mask0))
        h1 = torch.tanh(self.fc1(x * self.mask1))
        h = torch.cat([h0, h1], dim=-1)
        return self.readout(h)


class SingleRouteRNN(nn.Module):
    """Single pathway; hidden size = 2 * hidden_per_module for rough width match."""

    def __init__(self, hidden_per_module: int = 32):
        super().__init__()
        h = 2 * hidden_per_module
        self.fc = nn.Linear(12, h, bias=True)
        self.readout = nn.Linear(h, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc(x))
        return self.readout(h)


def holton_probe_loss(
    pred: torch.Tensor, label_x: torch.Tensor, label_y: torch.Tensor, feature_probe: torch.Tensor
) -> torch.Tensor:
    """MSE on cos/sin for the probed season; rows already store the matching target."""
    xy = torch.stack([label_x, label_y], dim=1)
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    n = 0
    for i in range(pred.shape[0]):
        p = int(feature_probe[i].item())
        if p == 0:
            loss = loss + F.mse_loss(pred[i, :2], xy[i])
            n += 1
        elif p == 1:
            loss = loss + F.mse_loss(pred[i, 2:4], xy[i])
            n += 1
    if n == 0:
        return loss
    return loss / n


def mean_mse_over_batches(
    model: nn.Module,
    input_b: torch.Tensor,
    probe_b: torch.Tensor,
    lx_b: torch.Tensor,
    ly_b: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    xy = torch.stack([lx_b, ly_b], dim=1)
    with torch.no_grad():
        for start in range(0, input_b.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            pred = model(input_b[sl])
            for i in range(pred.shape[0]):
                gi = start + i
                p = int(probe_b[gi].item())
                if p == 0:
                    total += F.mse_loss(pred[i, :2], xy[gi], reduction="sum").item()
                else:
                    total += F.mse_loss(pred[i, 2:4], xy[gi], reduction="sum").item()
                n += 1
    return total / max(n, 1)
