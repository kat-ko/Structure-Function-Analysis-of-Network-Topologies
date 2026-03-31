"""
RNN initialization scaling for A1-B-A2 modular RNN (init_scale knob).
Used as experimental lever; rich/lazy is determined post hoc from dynamics (e.g. arXiv:2310.08513).
"""
import torch


def apply_init_scale(network, scale, scope="global"):
    """
    Apply init scaling on RNN wrapper parameters.

    Parameters
    ----------
    network : TwoModuleRNNWrapper
        Wrapper containing a `community` module with named parameters.
    scale : float
        Multiplicative scale applied in-place after model construction.
    scope : {"global", "input_only"}, optional
        - "global": scale all trainable parameters (legacy/default behavior).
        - "input_only": scale only input-to-hidden matrices (`weight_ih*`),
          i.e., input pathway while leaving recurrent/readout weights unchanged.
    """
    if scope not in ("global", "input_only"):
        raise ValueError(f"Unsupported init scope {scope!r}. Use 'global' or 'input_only'.")

    for name, param in network.community.named_parameters():
        if not param.requires_grad:
            continue
        if scope == "global":
            param.data.mul_(scale)
        elif scope == "input_only":
            # PyTorch RNN/GRU convention: input-to-hidden parameters are named weight_ih*
            if "weight_ih" in name:
                param.data.mul_(scale)
