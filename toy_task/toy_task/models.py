"""Dense and modular Elman RNNs, aligned with ``a1b2_modular`` (Section 3).

This is the a1b2-aligned reimplementation. It follows
``a1b2_modular/a1b2/models/community.py`` closely, with two deliberate departures:

* **Inter-module communication** is optional via ``comms_bandwidth`` on modular
  architectures: a second ``comms`` RNN pathway (off-diagonal recurrent mask,
  scaled by bandwidth) sums with the block-diagonal ``core`` pathway, matching
  a1b2's dual-pathway layout. At ``comms_bandwidth=0`` only ``core`` exists.
* **Feature routing** (``modular_feature_routed``) is a toy_task-only input scheme
  with no a1b2 analog; it is expressed purely through the input mask.

Faithful to a1b2 in every other respect:

* a single ``nn.RNN`` (``bias=False``, tanh) over the module-replicated input;
* recurrent weights masked block-diagonal with the diagonal **self-connections
  removed** (``1 - I`` per block), exactly like a1b2's ``rec_mask``;
* input weights masked block-diagonal (a1b2 ``input_mask`` with ``common_input=False``);
* a block-diagonal masked readout whose per-module outputs are **summed**
  (a1b2 ``Readout`` with ``common_readout`` semantics).

All masks are applied via :func:`torch.nn.utils.parametrize.register_parametrization`
with a :class:`_MaskedWeight`, mirroring a1b2's ``Masked_weight``.

Contract (unchanged for downstream code)::

    out, hidden = model(x, task_id=...)

``x`` is ``(batch, d_x)``; ``out`` is ``(batch, d_y)``; ``hidden`` is the
final-timestep hidden state ``(batch, H)`` (modules concatenated). The same ``x`` is
fed for ``seq_len`` timesteps as a length-``seq_len`` sequence; the hidden state
starts at zero each call; readout/loss use the final timestep only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

from .config import (
    CONSTANTS,
    ARCH_DENSE,
    ARCH_MODULAR_SHARED,
    ARCH_MODULAR_FEATURE_ROUTED,
    ARCH_MODULAR_TASK_ROUTED,
    OFF_FREEZE,
    OFF_READOUT_COORD,
)
from .environment import feature_routing_slices


class _MaskedWeight(nn.Module):
    """Parametrization that masks a weight tensor (a1b2 ``Masked_weight``)."""

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.mask


# --------------------------------------------------------------------- masks
def _recurrent_mask(n_modules: int, per: int) -> torch.Tensor:
    """Block-diagonal recurrent mask; diagonal blocks are ``1 - I`` (no self-loops)."""
    H = n_modules * per
    mask = torch.zeros(H, H)
    no_self = 1.0 - torch.eye(per)
    for m in range(n_modules):
        mask[m * per:(m + 1) * per, m * per:(m + 1) * per] = no_self
    return mask


def _comms_recurrent_mask(n_modules: int, per: int, bandwidth: float) -> torch.Tensor:
    """Off-diagonal recurrent blocks scaled by ``bandwidth`` (a1b2 ``comms_mask``)."""
    H = n_modules * per
    mask = torch.zeros(H, H)
    for i in range(n_modules):
        for j in range(n_modules):
            if i != j:
                mask[i * per:(i + 1) * per, j * per:(j + 1) * per] = bandwidth
    return mask


def _input_mask(arch: str, n_modules: int, per: int, d_obs: int) -> torch.Tensor:
    """Block-diagonal input mask over the module-replicated input (size ``d_obs*n_modules``).

    ``modular_feature_routed`` additionally restricts each module's block to its
    contiguous feature slice.
    """
    in_total = d_obs * n_modules
    mask = torch.zeros(n_modules * per, in_total)
    if n_modules == 1:
        mask[:] = 1.0
        return mask
    slices = (
        feature_routing_slices(d_obs, n_modules)
        if arch == ARCH_MODULAR_FEATURE_ROUTED
        else None
    )
    for m in range(n_modules):
        block = torch.zeros(per, d_obs)
        if slices is not None:
            block[:, slices[m]] = 1.0
        else:
            block[:] = 1.0
        mask[m * per:(m + 1) * per, m * d_obs:(m + 1) * d_obs] = block
    return mask


def _readout_mask(n_modules: int, d_out: int, per: int) -> torch.Tensor:
    """Block-diagonal readout mask: module ``m``'s hidden block -> its own output block."""
    mask = torch.zeros(n_modules * d_out, n_modules * per)
    for m in range(n_modules):
        mask[m * d_out:(m + 1) * d_out, m * per:(m + 1) * per] = 1.0
    return mask


class ToyRNN(nn.Module):
    """a1b2-aligned single-layer Elman RNN covering all four architectures.

    Parameters
    ----------
    arch : str
        One of the architecture identifiers in ``config.ARCHITECTURES``.
    hidden_size : int
        Total hidden units ``H`` (modular variants use ``H / n_modules`` per module).
    off_module_policy : str
        Only used by ``modular_task_routed``: ``"freeze"`` or ``"readout_coord"``.
    """

    def __init__(self, arch: str, hidden_size: int, off_module_policy: str = OFF_FREEZE,
                 comms_bandwidth: float = 0.0):
        super().__init__()
        c = CONSTANTS
        self.arch = arch
        self.off_module_policy = off_module_policy
        self.comms_bandwidth = comms_bandwidth
        self.hidden_size = hidden_size
        self.seq_len = c.seq_len
        self.d_obs = c.d_obs
        self.d_out = c.d_out

        self.n_modules = 1 if arch == ARCH_DENSE else c.n_modules
        if hidden_size % self.n_modules != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by n_modules {self.n_modules}"
            )
        self.per = hidden_size // self.n_modules
        self.in_total = self.d_obs * self.n_modules
        input_mask = _input_mask(arch, self.n_modules, self.per, self.d_obs)

        # Recurrent core: a1b2's block-diagonal pathway.
        self.core = nn.RNN(
            input_size=self.in_total,
            hidden_size=hidden_size,
            num_layers=1,
            bias=False,
            nonlinearity="tanh",
            batch_first=False,
        )
        register_parametrization(self.core, "weight_ih_l0", _MaskedWeight(input_mask))
        register_parametrization(
            self.core, "weight_hh_l0",
            _MaskedWeight(_recurrent_mask(self.n_modules, self.per)),
        )

        # Optional comms pathway (modular archs only; omitted at bandwidth=0).
        self.comms: nn.RNN | None = None
        if self.n_modules > 1 and comms_bandwidth > 0.0:
            self.comms = nn.RNN(
                input_size=self.in_total,
                hidden_size=hidden_size,
                num_layers=1,
                bias=False,
                nonlinearity="tanh",
                batch_first=False,
            )
            register_parametrization(self.comms, "weight_ih_l0", _MaskedWeight(input_mask))
            register_parametrization(
                self.comms, "weight_hh_l0",
                _MaskedWeight(_comms_recurrent_mask(self.n_modules, self.per, comms_bandwidth)),
            )

        # Readout: a1b2 block-diagonal masked Linear, per-module outputs summed.
        # ``readout_coord`` keeps one such readout head per task (toy_task extension).
        self._readout_coord = (
            arch == ARCH_MODULAR_TASK_ROUTED and off_module_policy == OFF_READOUT_COORD
        )
        self._n_heads = self.n_modules if self._readout_coord else 1
        if self._n_heads > 1:
            self.readout = nn.ModuleList(
                [self._make_readout_head() for _ in range(self._n_heads)]
            )
        else:
            self.readout = self._make_readout_head()

    # ------------------------------------------------------------- building
    def _make_readout_head(self) -> nn.Linear:
        head = nn.Linear(self.hidden_size, self.n_modules * self.d_out)
        if self.n_modules > 1:
            register_parametrization(
                head, "weight",
                _MaskedWeight(_readout_mask(self.n_modules, self.d_out, self.per)),
            )
        return head

    # --------------------------------------------------------------- inputs
    def _module_replicated_input(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Replicate ``x`` once per module (a1b2 ``x.repeat(..., n_modules)``).

        For ``task_routed`` + ``freeze`` the inactive module's copy is zeroed at
        runtime (a1b2 ``_routed_input``); all other schemes replicate unchanged.
        """
        if self.n_modules == 1:
            return x
        if self.arch == ARCH_MODULAR_TASK_ROUTED and self.off_module_policy == OFF_FREEZE:
            parts = [
                x if m == task_id else torch.zeros_like(x)
                for m in range(self.n_modules)
            ]
            return torch.cat(parts, dim=1)
        return x.repeat(1, self.n_modules)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, task_id: int = 0):
        if x.dim() != 2:
            raise ValueError(f"expected x of shape (batch, d_obs); got {tuple(x.shape)}")
        batch = x.shape[0]
        inp = self._module_replicated_input(x, task_id)        # (batch, in_total)
        seq = inp.unsqueeze(0).expand(self.seq_len, batch, self.in_total)
        core_out, _ = self.core(seq)                           # (seq, batch, H)
        if self.comms is not None:
            comms_out, _ = self.comms(seq)
            out_seq = core_out + comms_out
        else:
            out_seq = core_out
        hidden = out_seq[-1]                                   # (batch, H)

        head = self.readout[task_id] if self._n_heads > 1 else self.readout
        out_full = head(hidden)                                # (batch, d_out*n_modules)
        if self.n_modules > 1:
            out = out_full.view(batch, self.n_modules, self.d_out).sum(dim=1)
        else:
            out = out_full
        return out, hidden

    # ----------------------------------------------------- introspection API
    @property
    def weight_ih(self) -> nn.Parameter:
        """Leaf input-weight tensor (pre-mask ``original``); gradients land here."""
        return self.core.parametrizations.weight_ih_l0.original

    @property
    def weight_hh(self) -> nn.Parameter:
        """Leaf recurrent-weight tensor (pre-mask ``original``)."""
        return self.core.parametrizations.weight_hh_l0.original

    def module_slice(self, m: int) -> slice:
        """Hidden / per-module row slice for module ``m``."""
        return slice(m * self.per, (m + 1) * self.per)

    def effective_weight_ih(self) -> torch.Tensor:
        """Masked input weight actually used in the forward pass."""
        return self.core.weight_ih_l0


def build_model(arch: str, hidden_size: int, off_module_policy: str = OFF_FREEZE,
                seed: int | None = None, comms_bandwidth: float = 0.0) -> ToyRNN:
    """Construct a :class:`ToyRNN`, optionally seeding torch for reproducible init."""
    if seed is not None:
        torch.manual_seed(seed)
    return ToyRNN(
        arch=arch,
        hidden_size=hidden_size,
        off_module_policy=off_module_policy,
        comms_bandwidth=comms_bandwidth,
    )
