"""
RNN initialization scaling for A1-B-A2 modular RNN (init_scale knob).
Used as experimental lever; rich/lazy is determined post hoc from dynamics (e.g. arXiv:2310.08513).
"""
import torch


def _iter_scaled_parameters(network, policy):
    """Yield (name, param) pairs in community according to scaling policy."""
    for name, param in network.community.named_parameters():
        if not param.requires_grad:
            continue
        if policy == "full":
            yield name, param
        elif policy == "exclude_readout":
            if not name.startswith("readout."):
                yield name, param
        else:
            raise ValueError(
                f"Unknown init_scale_policy '{policy}'. "
                "Supported: ['full', 'exclude_readout']"
            )


def apply_init_scale_policy(network, scale, policy="full"):
    """
    Scale trainable community parameters by `scale` using selected policy.

    Policies
    --------
    full:
        Scale all trainable params in `network.community`.
    exclude_readout:
        Scale all trainable params except `network.community.readout.*`.
    """
    for _, param in _iter_scaled_parameters(network, policy):
        param.data.mul_(scale)


def apply_init_scale(network, scale):
    """
    Backward-compatible helper: equivalent to policy='full'.
    """
    apply_init_scale_policy(network, scale, policy="full")
