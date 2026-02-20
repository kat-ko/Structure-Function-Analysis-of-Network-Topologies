"""
Plotting utilities for dynspec-style analyses (masks, metrics, behavior).
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from a1b2.analysis.retraining import global_diff_metric, diff_metric, metric_norm_acc


def set_style():
    """Optionally load style sheet; no-op if file not present."""
    try:
        file_path = os.path.realpath(__file__).replace("plotting.py", "style_sheet.mplstyle")
        if os.path.isfile(file_path):
            plt.style.use(file_path)
    except Exception:
        pass


def single_filter(data, key, value):
    if key[0] == "!":
        if value is None:
            return ~data[key[1:]].isnull()
        return data[key[1:]] != value
    if value is None:
        return data[key].isnull()
    return data[key] == value


def filter_data(data, v_params):
    data = data.copy()
    filters = []
    for key, value in v_params.items():
        if key in data.columns or (key[0] == "!" and key[1:] in data.columns):
            if isinstance(value, list):
                f = np.sum([single_filter(data, key, v) for v in value], axis=0).astype(bool)
            else:
                f = single_filter(data, key, value)
            filters.append(f)
    if not filters:
        return data, np.ones(len(data), dtype=bool)
    filter_ = np.prod(filters, axis=0).astype(bool)
    data = data[filter_]
    return data, filter_


def plot_model_masks(experiment, plot_input=False):
    """Plot recurrent and connection masks for each model in the experiment."""
    n_models = len(experiment.models)
    n1 = int(np.sqrt(n_models))
    n2 = int(np.ceil(np.sqrt(n_models)))
    if n1 * n2 < n_models:
        n2 += 1

    fig = plt.figure(
        figsize=(n2 * 2, n1 * 2 * (1 + plot_input * 0.3)), constrained_layout=True
    )
    subfigs = fig.subfigures(1 + plot_input, height_ratios=[1, 0.3][: (1 + plot_input)])
    if not plot_input:
        subfigs = np.array([subfigs])

    subfigs[0].suptitle("Recurrent + Connections Masks")
    axs = subfigs[0].subplots(n1, n2)
    if n_models == 1:
        axs = np.array([axs]).T

    for ax, model in zip(axs.flatten(), experiment.models):
        h = model.n_modules * model.hidden_size
        combined = (
            model.masks["comms_mask"][:h].cpu().numpy()
            + model.masks["rec_mask"][:h].cpu().numpy()
        )
        ax.imshow(combined, vmin=0, vmax=1)
        ax.set_title(f'n = {model.hidden_size}, p = {model.sparsity}')

    if plot_input:
        subfigs[1].suptitle("Input Masks")
        axs = subfigs[1].subplots(n1, n2)
        if n_models == 1:
            axs = np.array([axs]).T
        for ax, model in zip(axs.flatten(), experiment.models):
            h = model.n_modules * model.hidden_size
            ax.imshow(
                model.masks["input_mask"].cpu().numpy()[:h, :],
                aspect="auto",
                vmin=0,
                vmax=1,
            )
    return fig
