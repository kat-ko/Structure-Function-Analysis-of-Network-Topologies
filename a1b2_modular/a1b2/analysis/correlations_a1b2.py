"""
Fixed-information correlation metric for A1B2 two-module RNN (offline from npz).
Bená-style: fix one factor (feature or stimulus), permute other; within-module self-correlation; normalize.
"""
import numpy as np
from scipy.stats import pearsonr


def randperm_no_fixed(n):
    """Random permutation with no fixed points (for baseline self-correlation)."""
    perm = np.random.permutation(n)
    if n > 4 and np.any(perm == np.arange(n)):
        return randperm_no_fixed(n)
    return perm


def fixed_information_data_a1b2(hiddens_per_module, probes, stim_index, fixed_factor, permute_other=True):
    """
    Build index arrays for "fix one factor, permute other" for correlation metric.
    hiddens_per_module : (n_trials, n_modules, hidden_size)
    probes : (n_trials,) 0 or 1
    stim_index : (n_trials,) stimulus id
    fixed_factor : "feature" or "stimulus"
    permute_other : if True, permute the non-fixed dimension
    Returns structure to index into hiddens: for each fixed value, (indices_original, indices_permuted).
    For "feature": two groups (fix feature 0, fix feature 1); each group has (orig_idx, perm_idx) over trials.
    """
    n_trials = hiddens_per_module.shape[0]
    if fixed_factor == "feature":
        # Subset by probes == 0 and probes == 1; within each subset permute trial order
        out = []
        for f in range(2):
            idx = np.where(probes == f)[0]
            if len(idx) < 2:
                out.append((idx, idx))
                continue
            perm = np.random.permutation(len(idx))
            if permute_other:
                perm_idx = idx[perm]
            else:
                perm_idx = idx
            out.append((idx, perm_idx))
        return out
    elif fixed_factor == "stimulus":
        # Group by stim_index; for each stimulus, permute trial order (or probes)
        uniq = np.unique(stim_index)
        out = []
        for s in uniq:
            idx = np.where(stim_index == s)[0]
            if len(idx) < 2:
                out.append((idx, idx))
                continue
            perm = np.random.permutation(len(idx))
            if permute_other:
                perm_idx = idx[perm]
            else:
                perm_idx = idx
            out.append((idx, perm_idx))
        return out
    else:
        raise ValueError('fixed_factor must be "feature" or "stimulus"')


def get_correlation_a1b2(hiddens_per_module, perm, corr_func="pearsonr"):
    """
    Within-module self-correlation: for each module, corr(h[..., m, :], h[perm, m, :]).
    hiddens_per_module : (n_trials, n_modules, hidden_size)
    perm : (n_trials,) permutation indices
    Returns : (n_modules,) correlation per module.
    """
    n_trials, n_modules, hidden_size = hiddens_per_module.shape
    if corr_func == "pearsonr":
        corrs = []
        for m in range(n_modules):
            h = hiddens_per_module[:, m, :]  # (n_trials, hidden_size)
            h_perm = hiddens_per_module[perm, m, :]
            r = np.corrcoef(h.reshape(n_trials, -1), h_perm.reshape(n_trials, -1))[0, 1]
            if np.isnan(r):
                r = 0.0
            corrs.append(r)
        return np.array(corrs)
    elif corr_func == "cka":
        from a1b2.analysis.correlations import CKA
        cka_fn = CKA().linear_CKA
        corrs = []
        for m in range(n_modules):
            h = hiddens_per_module[:, m, :]
            h_perm = hiddens_per_module[perm, m, :]
            corrs.append(cka_fn(h, h_perm))
        return np.array(corrs)
    else:
        raise ValueError('corr_func must be "pearsonr" or "cka"')


def compute_correlation_metric_a1b2(participant_data, n_samples=10, fixed_factor="feature", corr_func="pearsonr"):
    """
    Offline correlation metric from one participant's npz-like data.
    participant_data : dict with keys hiddens_per_module (n_phase, n_trials, n_mod, h),
                       probes (n_phase, n_trials), inputs (n_phase, n_trials, dim_input) or stim_index.
    We flatten phase and trials to get (N, n_mod, h).
    Returns dict with base_correlations, correlations (per fixed value), norm_correlations, and specialization scalar.
    """
    hiddens = participant_data["hiddens_per_module"]
    probes = participant_data["probes"]
    if "stim_index" in participant_data:
        stim_index = participant_data["stim_index"]
    else:
        inputs = participant_data["inputs"]
        stim_index = np.argmax(inputs, axis=-1)
    # Flatten phase and trials
    if hiddens.ndim == 4:
        n_phase, n_trials, n_mod, h_size = hiddens.shape
        hiddens = hiddens.reshape(-1, n_mod, h_size)
        probes = probes.reshape(-1)
        if stim_index.ndim == 2:
            stim_index = stim_index.reshape(-1)
        elif stim_index.ndim == 3:
            stim_index = np.argmax(stim_index, axis=-1).reshape(-1)
    n_trials = hiddens.shape[0]
    base_correlations_list = []
    corr_fix0_list = []
    corr_fix1_list = []
    for _ in range(n_samples):
        perm = randperm_no_fixed(n_trials)
        base_correlations_list.append(get_correlation_a1b2(hiddens, perm, corr_func))
        idx0 = np.where(probes == 0)[0]
        idx1 = np.where(probes == 1)[0]
        if len(idx0) >= 2:
            p0 = randperm_no_fixed(len(idx0))
            c0 = get_correlation_a1b2(hiddens[idx0], p0, corr_func)
        else:
            c0 = np.zeros(hiddens.shape[1])
        if len(idx1) >= 2:
            p1 = randperm_no_fixed(len(idx1))
            c1 = get_correlation_a1b2(hiddens[idx1], p1, corr_func)
        else:
            c1 = np.zeros(hiddens.shape[1])
        corr_fix0_list.append(c0)
        corr_fix1_list.append(c1)
    base = np.mean(base_correlations_list, axis=0)
    corr0 = np.mean(corr_fix0_list, axis=0)
    corr1 = np.mean(corr_fix1_list, axis=0)
    denom = 1 - base
    denom[denom <= 0] = 1e-8
    norm0 = (corr0 - base) / denom
    norm1 = (corr1 - base) / denom
    norm0 = np.clip(norm0, 0.0, 1.0)
    norm1 = np.clip(norm1, 0.0, 1.0)
    scalar = correlation_specialization_scalar(norm0, norm1)
    return {
        "base_correlations": base,
        "correlations_fix_feature0": corr0,
        "correlations_fix_feature1": corr1,
        "norm_correlations_fix0": norm0,
        "norm_correlations_fix1": norm1,
        "correlation_specialization": scalar,
    }


def _diff_metric(pair):
    a, b = float(pair[0]), float(pair[1])
    s = a + b
    if s <= 0:
        return 0.0
    return (a - b) / s


def _global_diff_metric(metric_task0, metric_task1):
    return abs(_diff_metric(metric_task0) - _diff_metric(metric_task1)) / 2


def correlation_specialization_scalar(norm_corr_fix0, norm_corr_fix1):
    """
    Turn two normalized correlation vectors (per module) into one specialization scalar.
    High when module 0 tracks feature 0 and module 1 tracks feature 1 (or vice versa).
    """
    if norm_corr_fix0.shape[0] < 2 and norm_corr_fix1.shape[0] < 2:
        return 0.0
    m0_0 = float(norm_corr_fix0[0])
    m1_0 = float(norm_corr_fix0[1] if norm_corr_fix0.shape[0] > 1 else norm_corr_fix0[0])
    m0_1 = float(norm_corr_fix1[0])
    m1_1 = float(norm_corr_fix1[1] if norm_corr_fix1.shape[0] > 1 else norm_corr_fix1[0])
    return _global_diff_metric((m0_0, m1_0), (m0_1, m1_1))
