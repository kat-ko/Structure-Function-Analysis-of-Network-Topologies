"""
Script to preprocess participant data and apply exclusion criteria.
"""
import os
import sys

# Ensure project root is on path when running from scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from a1b2.data.preprocessing import (
    load_participant_data,
    add_computed_columns,
    exclude_participants,
    add_regressors,
)
from a1b2.data.basic_funcs import set_seed


def main():
    SEED = 2024
    set_seed(SEED)

    data_root = os.path.join(_PROJECT_ROOT, 'data')
    DATA_FOLDER = os.path.join(data_root, 'participants', 'raw')
    OUTPUT_PATH = os.path.join(data_root, 'participants', 'trial_df.csv')

    study1_batches = ['study1_same', 'study1_near', 'study1_far']
    study2_batches = ['study2_same', 'study2_near', 'study2_far']
    all_batches = study1_batches + study2_batches

    print("Loading data...")
    df = load_participant_data(DATA_FOLDER, all_batches)
    print(f'{df["participant"].nunique()} participants total')

    print("Adding computed columns...")
    df = add_computed_columns(df)

    print("Adding regressors and test trial information...")
    df = add_regressors(df)

    print("Applying exclusion criteria...")
    df, trial_df = exclude_participants(df)

    print(f'{trial_df["participant"].nunique()} participants post exclusion')
    print(f'{df["participant"].nunique() - trial_df["participant"].nunique()} dropped')

    grouped_counts = trial_df.groupby(
        ['participant', 'condition']
    ).size().reset_index(name='count')
    print("\nParticipants per condition:")
    print(grouped_counts.groupby('condition')['participant'].nunique())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nSaving processed data to {OUTPUT_PATH}")
    trial_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
