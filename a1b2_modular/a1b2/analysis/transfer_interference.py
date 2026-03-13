"""
Transfer-interference analysis: load ANN data, compute transfer metrics, PCA, principal angles.
"""
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

from a1b2.data.basic_funcs import wrap_to_pi


def load_ann_data(ann_folder, load_rnn_extra=False):
    """
    Load ANN simulation data from npz files and organize by condition.

    Args:
        ann_folder (str): Path to folder containing simulation results
            (e.g., 'data/simulations/rich_50')
        load_rnn_extra (bool): If True, also load RNN-specific keys when present in
            the npz: hiddens_per_module, hiddens_post_phase_*_trajectory,
            hiddens_post_phase_*_per_module. Default False.

    Returns:
        dict: Dictionary with keys 'same', 'near', 'far' containing simulation data
              for each condition
    """
    condition_data = {
        'same': [],
        'near': [],
        'far': []
    }

    rnn_extra_keys = [
        'hiddens_per_module',
        'hiddens_post_phase_0_trajectory', 'hiddens_post_phase_1_trajectory',
        'hiddens_post_phase_2_trajectory',
        'hiddens_post_phase_0_per_module', 'hiddens_post_phase_1_per_module',
        'hiddens_post_phase_2_per_module',
        'hiddens_post_phase_0_core_per_module', 'hiddens_post_phase_1_core_per_module',
        'hiddens_post_phase_2_core_per_module',
        'hiddens_post_phase_0_comms_per_module', 'hiddens_post_phase_1_comms_per_module',
        'hiddens_post_phase_2_comms_per_module',
    ]

    for file_name in os.listdir(ann_folder):
        if not file_name.endswith('.npz'):
            continue

        file_path = os.path.join(ann_folder, file_name)

        for condition in condition_data.keys():
            if condition in file_name:
                with np.load(file_path, allow_pickle=True) as data:
                    entry = {
                        'participant': file_name.replace('.npz', ''),
                        'predictions': data['predictions'],
                        'labels': data['labels'],
                        'accuracy': data['accuracy'],
                        'losses': data['losses'],
                        'test_stim': data['test_stim'],
                        'hiddens_post_phase_0': data['hiddens_post_phase_0'],
                        'hiddens_post_phase_1': data['hiddens_post_phase_1'],
                    }
                    if 'hiddens_post_phase_2' in data:
                        entry['hiddens_post_phase_2'] = data['hiddens_post_phase_2']
                    if 'probes' in data:
                        entry['probes'] = data['probes']
                    if load_rnn_extra:
                        for key in rnn_extra_keys:
                            if key in data:
                                entry[key] = data[key]
                    condition_data[condition].append(entry)
                break

    return condition_data


def setup_task_parameters():
    """Define basic task parameters."""
    return {
        "nStim_perTask": 6,
        "schedules": ['same', 'near', 'far'],
        "schedule_names": ['same rule', 'near rule', 'far rule']
    }


def load_participant_data(data_folder):
    """Load participant data for ANN training."""
    df = pd.read_csv(os.path.join(data_folder, 'participants', 'trial_df.csv'))
    df.loc[df['task_section'] == 'B', 'test_trial'] = 0
    return df.loc[(df['task_section'] == 'A1') |
                  (df['task_section'] == 'B') |
                  (df['task_section'] == 'A2'), :]


def numpy_to_python(obj):
    """Convert numpy objects to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj


def generate_geometry_df(df, participant_to_copy, near_rule=np.pi/6, far_rule=np.pi):
    """
    Create phantom DataFrame where A training is matched, for geometry visualisation.
    """
    conditions = {'same': 0, 'near': near_rule, 'far': far_rule}
    combined_dfs = []

    for condition, rule_adjustment in conditions.items():
        temp_df = df[df['participant'] == participant_to_copy].copy()
        temp_df['participant'] = f"geom_sub_{condition}"
        temp_df['condition'] = condition
        if condition != 'same':
            temp_df = adjust_rule_and_feat_val(temp_df, rule_adjustment)
        combined_dfs.append(temp_df)

    return pd.concat(combined_dfs, ignore_index=True)


def adjust_rule_and_feat_val(df_subset, rule_adjustment):
    """Adjust B_rule and feat_val for circular wrapping."""
    df_copy = df_subset.copy()
    df_copy['B_rule'] = wrap_to_pi(df_copy['B_rule'] + rule_adjustment)
    mask = (df_copy['task_section'] == 'B') & (df_copy['feature_idx'] == 1)
    df_copy.loc[mask, 'feat_val'] = wrap_to_pi(df_copy.loc[mask, 'feat_val'] + rule_adjustment)
    return df_copy


def compute_transfer_anns(ann_data):
    """Calculate transfer/switch cost metrics for ANN data."""
    agg_data = []

    for schedule_name, schedule_data in ann_data.items():
        for subj in range(len(schedule_data)):
            A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy()
            B_accuracy = schedule_data[subj]['accuracy'][1, 1::2].copy()

            final_A1_acc = np.mean(A1_accuracy[-6:])
            initial_B_acc = np.mean(B_accuracy[0:6])

            error_diff = initial_B_acc - final_A1_acc

            agg_data.append({
                'participant': str(schedule_data[subj]['participant']),
                'condition': schedule_name,
                'error_diff': error_diff
            })

    return pd.DataFrame(agg_data)


def _collapse_repeated_loss(loss_1d, rtol=1e-5, return_start_indices=False):
    """Reduce per-trial loss (repeated per batch) to one value per batch.

    If return_start_indices is True, returns (values, start_indices) where
    start_indices[j] is the trial index of the first element of batch j.
    """
    if len(loss_1d) == 0:
        return (loss_1d, np.array([], dtype=np.intp)) if return_start_indices else loss_1d
    loss_1d = np.asarray(loss_1d, dtype=np.float64)
    change = np.r_[True, ~np.isclose(loss_1d[1:], loss_1d[:-1], rtol=rtol, equal_nan=True)]
    out = loss_1d[change]
    if return_start_indices:
        return out, np.nonzero(change)[0]
    return out


def analyze_training_loss(ann_data, save_path=None, feature_style='winter_everywhere'):
    """Analyze and optionally save training loss curves for all schedules.

    Loss is stored as one value per trial with the same batch loss repeated
    for each trial in the batch. We collapse to one value per batch, then
    filter by feature and training-only (test_stim==0) when probes are available.

    feature_style:
        - 'winter_everywhere': winter probe (probe==1) in all phases. A2 shows
          untrained feature → flat block. Default for interference interpretation.
        - 'trained_per_phase': winter in A1/B, summer (probe==0) in A2. A2 shows
          re-learning → descending (Holton-like).
        - 'summer_only': summer probe everywhere; matches Holton [1::2] when
          trial order aligns (A2 descending).

    Without probes, falls back to [1::2]. Returns phase_boundaries (b1, b2) in
    filtered index space.
    """
    results = {}

    for schedule_name in ['same', 'near', 'far']:
        schedule_data = ann_data[schedule_name]
        has_probes = 'probes' in schedule_data[0] and schedule_data[0]['probes'] is not None

        per_batch_losses = []
        phase_boundaries_list = []

        for subj in range(len(schedule_data)):
            arr = schedule_data[subj]['losses']
            probes = schedule_data[subj].get('probes')
            test_stim = schedule_data[subj].get('test_stim')

            if arr.ndim == 2:
                if has_probes and probes is not None and probes.ndim >= 2:
                    # Per-phase feature selection: winter=1, summer=0
                    phase_losses = []
                    n_train_per_phase = []
                    for p in range(arr.shape[0]):
                        vals, starts = _collapse_repeated_loss(arr[p, :], return_start_indices=True)
                        prb = probes[0, p, :] if probes.ndim == 3 else probes[p, :]
                        batch_probe = prb[starts]
                        if test_stim is not None:
                            tst = test_stim[0, p, :] if test_stim.ndim == 3 else test_stim[p, :]
                            batch_test = np.asarray(tst[starts], dtype=np.float64)
                            train_mask = (batch_test == 0)
                        else:
                            train_mask = np.ones(len(starts), dtype=bool)
                        if feature_style == 'winter_everywhere':
                            mask = (batch_probe == 1) & train_mask
                        elif feature_style == 'summer_only':
                            mask = (batch_probe == 0) & train_mask
                        elif feature_style == 'trained_per_phase':
                            # A1 and B: winter; A2: summer (trained in A2)
                            if p == 2:
                                mask = (batch_probe == 0) & train_mask
                            else:
                                mask = (batch_probe == 1) & train_mask
                        else:
                            raise ValueError(
                                "feature_style must be 'winter_everywhere', "
                                "'trained_per_phase', or 'summer_only'"
                            )
                        phase_losses.append(vals[mask])
                        n_train_per_phase.append(np.sum(mask))
                    collapsed = np.concatenate(phase_losses, axis=0)
                    b1 = n_train_per_phase[0]
                    b2 = n_train_per_phase[0] + n_train_per_phase[1]
                    phase_boundaries_list.append((b1, b2))
                else:
                    collapsed_per_phase = [_collapse_repeated_loss(arr[p, :]) for p in range(arr.shape[0])]
                    n_batches = [len(c) for c in collapsed_per_phase]
                    b1 = n_batches[0]
                    b2 = n_batches[0] + n_batches[1]
                    phase_boundaries_list.append((b1, b2))  # full batch space
                    collapsed = np.concatenate(collapsed_per_phase, axis=0)[1::2]
            else:
                flat_loss = np.concatenate(arr, axis=0)
                collapsed = _collapse_repeated_loss(flat_loss)
                if not has_probes:
                    collapsed = collapsed[1::2]
            per_batch_losses.append(collapsed)

        min_len = min(len(x) for x in per_batch_losses)
        sched_losses = np.full((len(per_batch_losses), min_len), np.nan)
        for i, coll in enumerate(per_batch_losses):
            sched_losses[i, :] = coll[:min_len]

        mean_sub = np.nanmean(sched_losses, axis=0)
        std_sub = np.nanstd(sched_losses, axis=0)
        sub_len = len(mean_sub)

        phase_boundaries = None
        if phase_boundaries_list:
            b1, b2 = phase_boundaries_list[0]
            if not has_probes:
                b1, b2 = b1 // 2, b2 // 2
            b1 = min(b1, sub_len)
            b2 = min(b2, sub_len)
            phase_boundaries = (b1, b2)

        results[schedule_name] = {
            'mean': mean_sub,
            'std': std_sub,
            'phase_boundaries': phase_boundaries,
        }

    return results


def compute_pca_components(ann_data, variance_threshold=0.99):
    """Compute number of PCA components for variance threshold in hidden representations."""
    results = {'participant': [], 'condition': [], 'task': [], 'n_pca': []}

    for schedule_name, schedule_data in ann_data.items():
        for subj in range(len(schedule_data)):
            A_hids = schedule_data[subj]['hiddens_post_phase_0']
            B_hids = schedule_data[subj]['hiddens_post_phase_1']

            pca_A_full = PCA().fit(A_hids)
            pca_B_full = PCA().fit(B_hids)

            n_components_A = np.argmax(np.cumsum(pca_A_full.explained_variance_ratio_) >= variance_threshold) + 1
            n_components_B = np.argmax(np.cumsum(pca_B_full.explained_variance_ratio_) >= variance_threshold) + 1

            participant_id = str(schedule_data[subj]['participant'])
            results['participant'].append(participant_id)
            results['condition'].append(schedule_name)
            results['task'].append('post A')
            results['n_pca'].append(n_components_A)
            results['participant'].append(participant_id)
            results['condition'].append(schedule_name)
            results['task'].append('post B')
            results['n_pca'].append(n_components_B)

            if 'hiddens_post_phase_2' in schedule_data[subj]:
                C_hids = schedule_data[subj]['hiddens_post_phase_2']
                pca_C_full = PCA().fit(C_hids)
                n_components_C = np.argmax(
                    np.cumsum(pca_C_full.explained_variance_ratio_) >= variance_threshold
                ) + 1
                results['participant'].append(participant_id)
                results['condition'].append(schedule_name)
                results['task'].append('post A2')
                results['n_pca'].append(n_components_C)

    return pd.DataFrame(results)


def get_principal_angles(ann_data):
    """Compute principal angles between A and B subspaces per participant/condition."""
    results = {'participant': [], 'condition': [], 'principal_angle_between': []}

    for schedule_name, schedule_data in ann_data.items():
        for subj in range(len(schedule_data)):
            A_hids = schedule_data[subj]['hiddens_post_phase_1'][0:6, :].copy()
            B_hids = schedule_data[subj]['hiddens_post_phase_1'][6:, :].copy()

            angle_between, _ = compute_principal_angle(A_hids, B_hids, n_components=2)

            results['participant'].append(str(schedule_data[subj]['participant']))
            results['condition'].append(schedule_name)
            results['principal_angle_between'].append(angle_between)

    return pd.DataFrame(results)


def compute_principal_angle(A_hids, B_hids, n_components=2):
    """Compute principal angles between subspaces spanned by A_hids and B_hids."""
    pca_A = PCA(n_components=n_components)
    pca_B = PCA(n_components=n_components)

    V_A = pca_A.fit_transform(A_hids)
    V_B = pca_B.fit_transform(B_hids)

    inner_product_matrix = np.dot(pca_A.components_, pca_B.components_.T)
    _, singular_values, _ = np.linalg.svd(inner_product_matrix)
    principal_angles = np.arccos(np.clip(singular_values, -1.0, 1.0))
    principal_angles_degrees = np.degrees(principal_angles)

    return principal_angles_degrees[0], principal_angles_degrees


def prepare_pca_single_task(hids):
    """Fit PCA on task hidden layer and transform."""
    pca = PCA(n_components=2)
    pca.fit(hids)
    task_transformed = pca.transform(hids)
    return pca, task_transformed


def project_onto_pca(pca, data):
    """Project data onto the PCA space defined by pca."""
    return pca.transform(data)


def get_hiddens(geom_results):
    """Extract post-A and post-B hiddens from geometry results (3 conditions)."""
    hiddens_postA = geom_results['hiddens_post_phase_0'][0, :, :]
    same_hiddens_postB = geom_results['hiddens_post_phase_1'][0, :, :]
    near_hiddens_postB = geom_results['hiddens_post_phase_1'][1, :, :]
    far_hiddens_postB = geom_results['hiddens_post_phase_1'][2, :, :]
    return hiddens_postA, same_hiddens_postB, near_hiddens_postB, far_hiddens_postB


def add_ann_metrics(rich_data, lazy_data, rich_group_params, lazy_group_params):
    """Aggregate ANN metrics for rich/lazy comparison with human group params."""
    ann_behav_data = []

    for dat_name, schedule_data, group_params in zip(
            ['rich', 'lazy'], [rich_data, lazy_data], [rich_group_params, lazy_group_params]):

        for subj in range(len(schedule_data)):
            A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy()
            B_accuracy = schedule_data[subj]['accuracy'][1, 1::2].copy()
            A2_accuracy = schedule_data[subj]['accuracy'][2, 1::2].copy()

            final_A1_acc = np.mean(A1_accuracy[-6:])
            initial_B_acc = np.mean(B_accuracy[0:6])
            A2_accuracy_mean = np.mean(A2_accuracy)

            transfer_error_diff = initial_B_acc - final_A1_acc
            retest_error_diff = A2_accuracy_mean - final_A1_acc

            summer_accuracy = np.mean(schedule_data[subj]['accuracy'][0, 0::2].copy())

            test_stim = schedule_data[subj]['test_stim'][0, 1::2].copy().astype(int)
            all_A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy()
            all_A1_accuracy[test_stim == 0] = np.nan
            generalisation_accuracy = np.nanmean(all_A1_accuracy)

            retest_int = 1 - group_params.loc[
                group_params['participant'] == str(schedule_data[subj]['participant']),
                'A_weight_A2'
            ].values[0].astype(np.float32)

            ann_behav_data.append({
                'group': dat_name,
                'participant': str(schedule_data[subj]['participant']),
                'initialB': initial_B_acc,
                'transfer_error_diff': transfer_error_diff,
                'retest_error_diff': retest_error_diff,
                'summer_accuracy': summer_accuracy,
                'generalisation_acc': generalisation_accuracy,
                'interference': retest_int
            })

    return pd.DataFrame(ann_behav_data)
