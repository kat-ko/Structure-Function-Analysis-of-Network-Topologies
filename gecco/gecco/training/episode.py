from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from gecco.models.routing_rnn import holton_probe_loss, mean_mse_over_batches


def apply_init_scale(model: nn.Module, scale: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)


def train_on_schedule(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
    *,
    steps: int,
    lr: float = 0.02,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    SGD-style minibatch training on a fixed cyclic A-B-... schedule tensor dict.

    Returns (train_mse_approx, val_mse) where val is last 15% of trials (no shuffle leak).
    """
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inp = tensors["input"].to(device)
    pr = tensors["feature_probe"].to(device)
    lx = tensors["label_x"].to(device)
    ly = tensors["label_y"].to(device)
    n = inp.shape[0]
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    cut = int(max(1, n * 0.85))
    perm_val = torch.arange(cut, n, device=device)

    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=rng, device=device)
        opt.zero_grad()
        pred = model(inp[idx])
        loss = holton_probe_loss(pred, lx[idx], ly[idx], pr[idx])
        loss.backward()
        opt.step()

    train_est = mean_mse_over_batches(model, inp[:cut], pr[:cut], lx[:cut], ly[:cut], batch_size=64)
    val = mean_mse_over_batches(model, inp[perm_val], pr[perm_val], lx[perm_val], ly[perm_val], batch_size=64)
    return float(train_est), float(val)
