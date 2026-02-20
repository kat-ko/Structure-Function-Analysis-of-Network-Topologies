"""
RNN initialization scaling for A1-B-A2 modular RNN (init_scale knob).
Used as experimental lever; rich/lazy is determined post hoc from dynamics (e.g. arXiv:2310.08513).
"""
import torch


def apply_init_scale(network, scale):
    """
    Scale all parameters of the RNN wrapper's community by `scale`.
    Call after construction to use as init-scale lever (no rich/lazy labels).
    """
    for name, param in network.community.named_parameters():
        if param.requires_grad:
            param.data.mul_(scale)
