# Fits von Mises distributions to participant data or simulation data.
# For participant data:
#   python scripts/03_fit_vonmises.py participants
# For simulation data:
#   python scripts/03_fit_vonmises.py simulations --sim-name rich_50
# Optional: specify base folder
#   python scripts/03_fit_vonmises.py participants --base-folder /path/to/project

import os
import sys
import json as _json
from pathlib import Path

# #region agent log
_log_path = "/home/kathrin/workspace/Structure-Function-Analysis-of-Network-Topologies/.cursor/debug.log"
_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
try:
    with open(_log_path, "a") as _f:
        _f.write(_json.dumps({"location": "03_fit_vonmises.py:pre_import", "message": "sys.path and root before any a1b2 import", "data": {"sys_path_first5": list(sys.path)[:5], "script_dir": str(_script_dir), "root": str(_root), "root_in_sys_path": str(_root) in sys.path}, "hypothesisId": "H1", "timestamp": __import__("time").time()}) + "\n")
except Exception:
    pass
# #endregion

# Ensure a1b2 is importable when run from a1b2_modular (scripts/ is on path, not parent)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# #region agent log
try:
    with open(_log_path, "a") as _f:
        _f.write(_json.dumps({"location": "03_fit_vonmises.py:post_insert", "message": "after sys.path insert", "data": {"root_in_sys_path": str(_root) in sys.path}, "hypothesisId": "H4", "runId": "post-fix", "timestamp": __import__("time").time()}) + "\n")
except Exception:
    pass
# #endregion

import numpy as np
import pandas as pd
import argparse

from a1b2.models import vonmises as vm
from a1b2.data.basic_funcs import wrap_to_pi
from a1b2.analysis import participant as participant_module
from a1b2.analysis import transfer_interference as ann
from a1b2.analysis import stats


def fit_human_data(participant_data):
    """Fit von mises to human data and perform model comparison."""
    grouped_df = (participant_data.groupby(['participant', 'condition', 'study'])[['A_rule', 'B_rule']]
                  .mean()
                  .reset_index()
                  .query("condition in ['near', 'far']"))

    sections = {'mixture': ['A1', 'B', 'A2'], 'compare': ['B', 'A2']}
    parameters = {'mixture': ['A_weight', 'kappa'], 'compare': ['A_LL', 'B_LL']}

    for phase, phase_sections in sections.items():
        for section in phase_sections:
            for param in parameters[phase]:
                grouped_df[f'{param}_{section}'] = np.nan

    for p in grouped_df['participant'].unique():
        print(f'Processing participant {p}')
        p_data = participant_data[participant_data['participant'] == p]
        A_rule = p_data['A_rule'].iloc[0]
        B_rule = p_data['B_rule'].iloc[0]

        responses = {
            'A1': p_data.query("task_section == 'A1' and feature_idx == 1 and block > 0")['rule_applied'].values,
            'B': p_data.query("task_section == 'B' and feature_idx == 1 and block > 10")['rule_applied'].values,
            'A2': p_data.query("task_section == 'A2' and feature_idx == 1")['rule_applied'].values
        }

        for section in sections['mixture']:
            fit_results = vm.fit_mixture_model(responses[section], A_rule, B_rule)
            for param, value in fit_results.items():
                grouped_df.loc[grouped_df['participant'] == p, f'{param}_{section}'] = value

        for section in sections['compare']:
            comparison_results = vm.compare_models(responses[section], A_rule, B_rule)
            for param, value in comparison_results.items():
                grouped_df.loc[grouped_df['participant'] == p, f'{param}_{section}'] = value

    return grouped_df


def fit_ann_data(ann_data):
    """Fit von mises to ANN data."""
    n_stim = 6

    participants = []
    conditions = []
    for condition in ['near', 'far']:
        for i in range(len(ann_data[condition])):
            participants.append(str(ann_data[condition][i]['participant']))
            conditions.append(condition)

    grouped_df = pd.DataFrame({'participant': participants, 'condition': conditions})

    for section in ['A1', 'B', 'A2']:
        grouped_df[f'A_weight_{section}'] = np.nan
        grouped_df[f'kappa_{section}'] = np.nan

    for s_idx, schedule_data in enumerate([ann_data['near'], ann_data['far']]):
        for subj in range(len(schedule_data)):
            print(f'Processing ANN participant {subj}')

            ruleA = np.arctan2(schedule_data[subj]['labels'][0, 1, :][0], schedule_data[subj]['labels'][0, 1, :][1]) - \
                    np.arctan2(schedule_data[subj]['labels'][0, 0, :][0], schedule_data[subj]['labels'][0, 0, :][1])
            ruleB = np.arctan2(schedule_data[subj]['labels'][1, 1, :][0], schedule_data[subj]['labels'][1, 1, :][1]) - \
                    np.arctan2(schedule_data[subj]['labels'][1, 0, :][0], schedule_data[subj]['labels'][1, 0, :][1])

            responses = {}
            for task_section_idx, section in enumerate(['A1', 'B', 'A2']):
                summer_radians = np.arctan2(schedule_data[subj]['predictions'][task_section_idx, ::2, 0],
                                            schedule_data[subj]['predictions'][task_section_idx, ::2, 1])
                winter_radians = np.arctan2(schedule_data[subj]['predictions'][task_section_idx, 1::2, 2],
                                            schedule_data[subj]['predictions'][task_section_idx, 1::2, 3])
                response_angle = winter_radians - summer_radians
                response_angle = wrap_to_pi(response_angle)
                responses[section] = np.concatenate([response_angle[i:i + n_stim]
                                                      for i in range(0, len(response_angle), n_stim * 100)])

            participant_id = str(schedule_data[subj]['participant'])
            for section in ['A1', 'B', 'A2']:
                fit_results = vm.fit_mixture_model(responses[section], ruleA, ruleB)
                grouped_df.loc[grouped_df['participant'] == participant_id, f'A_weight_{section}'] = fit_results['A_weight']
                grouped_df.loc[grouped_df['participant'] == participant_id, f'kappa_{section}'] = fit_results['kappa']

    return grouped_df


def run_analysis(data_type, sim_name=None, base_folder='./', simulations_subdir='simulations'):
    """Run von Mises analysis for participants or simulations."""
    data_folder = os.path.join(base_folder, 'data')

    if data_type == 'participants':
        participant_data = pd.read_csv(os.path.join(data_folder, 'participants', 'trial_df.csv'))
        grouped_df = fit_human_data(participant_data)
        output_path = os.path.join(data_folder, 'participants', 'human_vonmises_fits.csv')
    elif data_type == 'simulations':
        if not sim_name:
            raise ValueError("Simulation name must be provided for simulation data")
        sim_root = os.path.join(data_folder, simulations_subdir)
        ann_data = ann.load_ann_data(os.path.join(sim_root, sim_name))
        grouped_df = fit_ann_data(ann_data)
        output_path = os.path.join(sim_root, f'{sim_name}_vonmises_fits.csv')
    else:
        raise ValueError("data_type must be 'participants' or 'simulations'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fit von Mises distributions to participant or simulation data')
    parser.add_argument('data_type', choices=['participants', 'simulations'], help='Type of data to analyze')
    parser.add_argument('--sim-name', type=str, help='Name of simulation folder (required if data_type is simulations)')
    parser.add_argument('--base-folder', type=str, default='./', help='Base project folder path (default: current directory)')
    parser.add_argument(
        '--simulations-subdir',
        type=str,
        default='simulations',
        help='Simulation subdirectory under data/ (e.g. simulations or simulations/primary_grid_ablations).',
    )
    args = parser.parse_args()
    run_analysis(args.data_type, args.sim_name, args.base_folder, args.simulations_subdir)


if __name__ == "__main__":
    main()
