import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black', width=.5)


def significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


def plot_transfer(data, var, condition_order, ylabel, xlim, ylim, yticks, schedule_colours, p_values,
                  addtests=1, markersize=1.5, scatter=True, figsize=[3 / 2.54, 4.5 / 2.54]):
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=data, x='condition', y=var, ax=ax,
                  order=condition_order, hue_order=condition_order,
                  palette=schedule_colours, hue='condition', legend=False,
                  alpha=0.3, size=2.5, linewidth=0, jitter=True, zorder=1)
    sns.pointplot(data=data, x='condition', y=var, ax=ax,
                  order=condition_order, palette='dark:k', hue='condition',
                  markers='', errorbar=('se'), linewidth=1, zorder=2)
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 2:
            line.set_xdata([xdata[0] - 0.1, xdata[1] - 0.1])
    sns.pointplot(data=data, x='condition', y=var, ax=ax,
                  order=condition_order, color='k', errorbar=None, markers='o', markersize=3.5,
                  linewidth=0.75, **{'markerfacecolor': 'white'}, zorder=3)
    if addtests:
        y_max = data[var].max() + 0.05
        for i, (pair, p_value) in enumerate(zip([(0, 1), (0, 2), (1, 2)], p_values)):
            if p_value < 0.05:
                x_coords = pair
                y_coord = y_max + (i * 0.1)
                ax.plot(x_coords, [y_coord + 0.05] * 2, color='black', linewidth=0.5)
                ax.text(np.mean(x_coords), y_coord, significance_stars(p_value), ha='center', va='bottom', fontsize=6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(condition_order, rotation=25)
    _style_axes(ax)
    return fig, ax


def plot_interference(param_df, A2_param, schedule_colours, figsize=[3 / 2.54, 4.5 / 2.54], ylabel='interference weight'):
    fig, ax = plt.subplots(figsize=figsize)
    param_df_tmp = param_df.loc[param_df['condition'] != 'same'].copy()
    param_df_tmp['A2_interference'] = 1 - param_df_tmp[A2_param]
    sns.stripplot(data=param_df_tmp, x='condition', y='A2_interference', ax=ax,
                  order=['near', 'far'], palette=schedule_colours[1:], hue='condition',
                  hue_order=['near', 'far'], legend=False, alpha=0.3, size=2.5, linewidth=0, jitter=True, zorder=1)
    sns.pointplot(data=param_df_tmp, x='condition', y='A2_interference', ax=ax,
                  order=['near', 'far'], palette='dark:k', hue='condition', legend=False,
                  markers='', errorbar=('se'), linewidth=1, zorder=2)
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 2:
            line.set_xdata([xdata[0] - 0.05, xdata[1] - 0.05])
    sns.pointplot(data=param_df_tmp, x='condition', y='A2_interference', ax=ax,
                  order=['near', 'far'], color='k', errorbar=None, markers='o', markersize=3.5,
                  linewidth=0.75, **{'markerfacecolor': 'white'}, zorder=3)
    ax.set_ylim([-.1, 1.1])
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0, 1], ['near', 'far'])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], va='center')
    _style_axes(ax)
    return fig, ax


def plot_accuracy_timecourse(trial_df, feature_idx, schedule_colours, condition_order, figsize=[7 / 2.54, 4 / 2.54]):
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(ax=ax, data=trial_df.loc[(trial_df['feature_idx'] == feature_idx) & (trial_df['task_section'] != 'A2'), :],
                 x='block', y='accuracy', hue='condition', errorbar='se', palette=schedule_colours,
                 hue_order=condition_order, linewidth=1)
    ax.axvline(10, linestyle='--', color='k', linewidth=0.5)
    ax.axhline(0.5, linestyle='-', color='grey', linewidth=5, alpha=0.2)
    ax.set_ylabel('winter accuracy' if feature_idx == 1 else 'summer accuracy')
    ax.set_xlabel('block')
    ax.set_yticks(np.arange(0.5, 1.1, 0.25), np.arange(0.5, 1.1, 0.25))
    ax.set_ylim([0.4, 1])
    ax.set_xticks(range(0, 21, 10))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center', bbox_to_anchor=(1.3, 0.5))
    _style_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_loss_curves(ann_data, schedule_name, schedule_colours, n_epochs=100, figsize=[3 / 2.54, 2 / 2.54]):
    schedule_data = ann_data[schedule_name]
    s_idx = ['same', 'near', 'far'].index(schedule_name)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(n_epochs * 6 * 10, color='k', linestyle='--', alpha=0.3)
    ax.axvline(2 * n_epochs * 6 * 10, color='k', linestyle='--', alpha=0.3)
    losses_arr = schedule_data[0]['losses']
    flattened_length = losses_arr.size if losses_arr.ndim == 2 else (np.asarray(losses_arr[0]).size * len(losses_arr))
    sched_losses = np.zeros((len(schedule_data), flattened_length))
    for subj in range(len(schedule_data)):
        arr = schedule_data[subj]['losses']
        flat_loss = np.ravel(arr) if arr.ndim == 2 else np.concatenate(arr, axis=0)
        sched_losses[subj, :] = flat_loss[:flattened_length]
    mean_losses = np.nanmean(sched_losses, axis=0)[1::2]
    std_losses = np.nanstd(sched_losses, axis=0)[1::2]
    x_values = np.arange(len(mean_losses))
    ax.plot(x_values, mean_losses, color=schedule_colours[s_idx], label=schedule_name, alpha=0.8, linewidth=1.5)
    ax.set_xticks(range(0, len(x_values), n_epochs * 6 * 10), ['A', 'B', 'A'])
    ax.set_yticks(np.arange(0, 0.51, 0.5))
    ax.set_yticklabels(np.arange(0, 0.51, 0.5))
    ax.set_xlabel("task")
    ax.set_ylabel("loss (MSE)")
    _style_axes(ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return fig, ax


def plot_pca_components(agg_df_long, task_colours, cm_conv=1/2.54):
    fig, ax = plt.subplots(figsize=[3*cm_conv, 3*cm_conv])
    _style_axes(ax)
    sns.barplot(ax=ax, data=agg_df_long, x='condition', y='n_pca', hue='task', palette=task_colours)
    ax.legend(title=None, fontsize=7)
    ax.set_ylabel('# PCA')
    ax.set_xlabel('')
    return fig, ax


def get_axis_limits(data):
    return {'xmin': data[:, 0].min(), 'xmax': data[:, 0].max(), 'ymin': data[:, 1].min(), 'ymax': data[:, 1].max()}


# Post A + Post B + Post A2 PCA geometry: Task A yellow → orange → red by phase;
# Task B light blue → dark blue → purple by phase.
POST_A_TASK_A_COLOUR = "#FDE047"
POST_B_TASK_A_COLOUR = "#F97316"
POST_A2_TASK_A_COLOUR = "#DC2626"
POST_A_TASK_B_COLOUR = "#7DD3FC"
POST_B_TASK_B_COLOUR = "#1D4ED8"
POST_A2_TASK_B_COLOUR = "#6D28D9"


def plot_2d_pca(ax, data, color, label):
    ax.plot(data[:, 0], data[:, 1], c=color, label=label, linestyle='-', marker='o', linewidth=1, markersize=6, markeredgewidth=0)
    ax.plot([data[-1, 0], data[0, 0]], [data[-1, 1], data[0, 1]], c=color, linewidth=1)


def plot_split_stim(ax, hiddens_pca, task_colours, lims):
    A_stim_hiddens = hiddens_pca[:6]
    B_stim_hiddens = hiddens_pca[6:]
    plot_2d_pca(ax, A_stim_hiddens, task_colours[0], 'Task A Stimuli')
    plot_2d_pca(ax, B_stim_hiddens, task_colours[1], 'Task B Stimuli')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_xlim(lims['xmin'] - .5, lims['xmax'] + .5)
    ax.set_ylim(lims['ymin'] - .5, lims['ymax'] + .5)
    ax.set_xticks([-1, 0, 1], [-1, 0, 1])
    ax.set_yticks([-1, 0, 1], [-1, 0, 1])


def plot_split_stim_postA_postB_postA2(ax, X2_A, X2_B, X2_A2, lims):
    """Six stimulus loops on shared 2D PCA axes (Post A, Post B, Post A2 × Task A/B)."""
    X2_A = np.asarray(X2_A)
    X2_B = np.asarray(X2_B)
    X2_A2 = np.asarray(X2_A2)
    plot_2d_pca(ax, X2_A[:6], POST_A_TASK_A_COLOUR, 'Post A — Task A')
    plot_2d_pca(ax, X2_A[6:], POST_A_TASK_B_COLOUR, 'Post A — Task B')
    plot_2d_pca(ax, X2_B[:6], POST_B_TASK_A_COLOUR, 'Post B — Task A')
    plot_2d_pca(ax, X2_B[6:], POST_B_TASK_B_COLOUR, 'Post B — Task B')
    plot_2d_pca(ax, X2_A2[:6], POST_A2_TASK_A_COLOUR, 'Post A2 — Task A')
    plot_2d_pca(ax, X2_A2[6:], POST_A2_TASK_B_COLOUR, 'Post A2 — Task B')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_xlim(lims['xmin'] - .5, lims['xmax'] + .5)
    ax.set_ylim(lims['ymin'] - .5, lims['ymax'] + .5)
    ax.set_xticks([-1, 0, 1], [-1, 0, 1])
    ax.set_yticks([-1, 0, 1], [-1, 0, 1])


def _plot_3d_stim_loop(ax, pts, colour, label=None, linewidth=1.2, alpha_line=0.55, scatter_s=28, scatter_alpha=0.9):
    pts = np.asarray(pts)
    if pts.shape[0] < 2:
        return None
    closed = np.vstack([pts, pts[0]])
    (line,) = ax.plot(
        closed[:, 0], closed[:, 1], closed[:, 2],
        color=colour, alpha=alpha_line, linewidth=linewidth, label=label,
    )
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=scatter_s, alpha=scatter_alpha,
               color=colour, edgecolors="none")
    return line


def plot_split_stim_postA_postB_postA2_3d(ax, X3_A, X3_B, X3_A2, labels_for_legend=False):
    """Six closed stimulus trajectories in 3D PCA space (shared axes per subplot)."""
    X3_A = np.asarray(X3_A)
    X3_B = np.asarray(X3_B)
    X3_A2 = np.asarray(X3_A2)
    handles = []
    specs = [
        (X3_A[:6], POST_A_TASK_A_COLOUR, 'Post A — Task A'),
        (X3_A[6:], POST_A_TASK_B_COLOUR, 'Post A — Task B'),
        (X3_B[:6], POST_B_TASK_A_COLOUR, 'Post B — Task A'),
        (X3_B[6:], POST_B_TASK_B_COLOUR, 'Post B — Task B'),
        (X3_A2[:6], POST_A2_TASK_A_COLOUR, 'Post A2 — Task A'),
        (X3_A2[6:], POST_A2_TASK_B_COLOUR, 'Post A2 — Task B'),
    ]
    for pts, colour, lab in specs:
        label = lab if labels_for_legend else None
        line = _plot_3d_stim_loop(ax, pts, colour, label=label)
        if line is not None and labels_for_legend:
            handles.append(line)
    return handles


def plot_near_hist(data, schedule_colours, figsize=[4.5 / 2.54, 3 / 2.54]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(1 - data['A_weight_A2'], color=schedule_colours[1], linewidth=0.5, edgecolor='k', alpha=0.5)
    ax.set_yticks(range(0, 41, 20))
    ax.set_ylim([0, 44])
    ax.set_xlabel('retest interference\n$\\it{p}$(Rule B)')
    _style_axes(ax)
    return fig, ax


def plot_id_1group(data, grouping, group_order, group_names, var, yticks, ytick_labs, ylim, ylab, colors, add_tests=0, p_value=np.nan, y_coord=np.nan, figsize=[3 / 2.54, 4.5 / 2.54]):
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=data, x=grouping, y=var, ax=ax, color=colors, order=group_order, legend=False,
                  alpha=0.3, size=3.5, linewidth=0, jitter=True, zorder=1)
    sns.pointplot(data=data, x=grouping, y=var, ax=ax, order=group_order, color='k', linestyle='-',
                  markers='', errorbar='se', linewidth=1, zorder=2)
    for line in ax.lines:
        if len(line.get_xdata()) == 2:
            line.set_xdata(line.get_xdata() - 0.05)
    sns.pointplot(data=data, x=grouping, y=var, ax=ax, order=group_order, color='k', errorbar=None,
                  markers='o', markersize=3.5, linewidth=0.75, linestyles="", markerfacecolor='white', zorder=3)
    _style_axes(ax)
    ax.set_xticks([0, 1], group_names)
    ax.set_xlabel('')
    ax.set_xlim([-0.5, 1.5])
    ax.set_yticks(yticks, ytick_labs)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)
    if add_tests and p_value < 0.05:
        ax.plot([0.2, 0.8], [y_coord] * 2, color='black', linewidth=0.5)
        ax.text(0.5, y_coord, significance_stars(p_value), ha='center', va='bottom', fontsize=6)
    return fig, ax


def plot_id_groups(data, grouping, group_order, group_names, var, yticks, ytick_labs, ylim, ylab, colors='grey', add_tests=0, p_value=np.nan, y_coord=np.nan, figsize=[6 / 2.54, 4.5 / 2.54]):
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=data, x=grouping, y=var, ax=ax, hue='ann', palette=colors, order=group_order, legend=False,
                  alpha=0.3, size=3.5, linewidth=0, jitter=True, zorder=1)
    sns.pointplot(data=data, x=grouping, y=var, ax=ax, order=group_order, color='k', linestyle='',
                  markers='', errorbar='se', linewidth=1, zorder=2)
    means = [np.mean(data.loc[data[grouping] == group, var]) for group in group_order]
    ax.plot([0, 1], means[:2], color='k')
    ax.plot([2, 3], means[2:], color='k')
    for line in ax.lines:
        if len(line.get_xdata()) == 2:
            line.set_xdata(line.get_xdata() - 0.05)
    sns.pointplot(data=data, x=grouping, y=var, ax=ax, order=group_order, color='k', errorbar=None,
                  markers='o', markersize=3.5, linewidth=0.75, linestyles="", markerfacecolor='white', zorder=3)
    _style_axes(ax)
    ax.set_xlabel('')
    ax.text(0.25, -0.25, 'participants', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.75, -0.25, 'ANNs', ha='center', va='center', transform=ax.transAxes)
    ax.set_yticks(yticks, ytick_labs)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)
    if add_tests and p_value < 0.05:
        ax.plot([0.2, 0.8], [y_coord] * 2, color='black', linewidth=0.5)
        ax.text(0.5, y_coord, significance_stars(p_value), ha='center', va='bottom', fontsize=6)
    return fig, ax
