"""Init-scale (gamma) manipulation for rich vs lazy regimes (Part III Section 18).

Mirrors ``a1b2_modular/a1b2/models/rnn_init.py``: after standard initialization,
multiply trainable parameters in place by ``gamma``. ``gamma`` is an experimental
lever; the rich/lazy label is assigned post hoc from dimensionality dynamics, not
from ``gamma`` itself.

The ``scope`` argument selects which parameters are scaled, matching a1b2's
``init_scope`` options:

* ``"all"`` (default)  -- every trainable parameter, **including** the readout.
* ``"no_readout"``     -- recurrent + input weights only; readout left at init.
* ``"input_only"``     -- only the input-to-hidden weights (``weight_ih``).
"""

from __future__ import annotations

import torch.nn as nn

from .config import (
    INIT_SCOPE_ALL,
    INIT_SCOPE_NO_READOUT,
    INIT_SCOPE_INPUT_ONLY,
    INIT_SCOPES,
)


def _is_readout_param(name: str) -> bool:
    return "readout" in name


def _is_input_param(name: str) -> bool:
    # Covers nn.RNN's ``weight_ih_l0`` and its parametrized form
    # ``core.parametrizations.weight_ih_l0.original``.
    return "weight_ih" in name


def apply_init_scale(model: nn.Module, gamma: float, scope: str = INIT_SCOPE_ALL) -> None:
    """Scale ``model`` parameters in place by ``gamma`` under the given ``scope``.

    A ``gamma`` of 1.0 is a no-op regardless of scope.
    """
    if scope not in INIT_SCOPES:
        raise ValueError(f"Unknown init scope {scope!r}; expected one of {INIT_SCOPES}")
    if gamma == 1.0:
        return
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if scope == INIT_SCOPE_NO_READOUT and _is_readout_param(name):
            continue
        if scope == INIT_SCOPE_INPUT_ONLY and not _is_input_param(name):
            continue
        param.data.mul_(gamma)
