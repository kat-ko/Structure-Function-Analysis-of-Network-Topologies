#!/usr/bin/env python3
"""
Script to generate geometry result files for all simulation configurations.

This script runs the geometry experiment for each configuration defined in
ann_experiments.json and saves the results as geom_results_{config_name}.npz files.

Usage:
    python scripts/generate_all_geometry_results.py
    python scripts/generate_all_geometry_results.py --configs lazy_50 rich_50  # Run specific configs only
"""

import os
import sys
import json
import argparse

# Get the script's directory and the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # transfer-interference directory

# Add project root to Python path so we can import src
sys.path.insert(0, PROJECT_ROOT)

# Import the function directly from the module
import importlib.util
spec = importlib.util.spec_from_file_location("run_simulations", 
    os.path.join(SCRIPT_DIR, "02_run_simulations.py"))
run_simulations_module = importlib.util.module_from_spec(spec)
sys.path.insert(0, SCRIPT_DIR)  # Add script dir for imports
spec.loader.exec_module(run_simulations_module)
run_geometry_experiment = run_simulations_module.run_geometry_experiment


def main():
    parser = argparse.ArgumentParser(description='Generate geometry results for all configurations')
    parser.add_argument('--configs', nargs='+', type=str, default=None,
                       help='Specific configurations to run (default: all)')
    parser.add_argument('--participant', type=str, default='study1_same_sub20',
                       help='Participant ID for geometry experiment')
    parser.add_argument('--base-folder', type=str, default=None,
                       help='Base project folder path (default: script parent directory)')
    
    args = parser.parse_args()
    
    # Use PROJECT_ROOT if base_folder not specified
    base_folder = args.base_folder if args.base_folder else PROJECT_ROOT
    
    # Load configuration file
    config_path = os.path.join(base_folder, 'src', 'models', 'ann_experiments.json')
    with open(config_path, 'r') as f:
        settings = json.load(f)
    
    # Get list of configurations to run
    if args.configs:
        configs_to_run = args.configs
        # Validate that all requested configs exist
        available_configs = [c['name'] for c in settings['conditions']]
        invalid_configs = [c for c in configs_to_run if c not in available_configs]
        if invalid_configs:
            print(f"Error: The following configurations were not found: {invalid_configs}")
            print(f"Available configurations: {available_configs}")
            sys.exit(1)
    else:
        configs_to_run = [c['name'] for c in settings['conditions']]
    
    print(f"Generating geometry results for {len(configs_to_run)} configurations:")
    print(f"  {', '.join(configs_to_run)}")
    print(f"Using participant: {args.participant}")
    print()
    
    # Run geometry experiment for each configuration
    successful = []
    failed = []
    
    for i, config_name in enumerate(configs_to_run, 1):
        print(f"[{i}/{len(configs_to_run)}] Processing {config_name}...")
        try:
            run_geometry_experiment(config_name, args.participant, base_folder)
            successful.append(config_name)
            print(f"✓ Successfully generated geometry results for {config_name}\n")
        except Exception as e:
            failed.append((config_name, str(e)))
            print(f"✗ Failed to generate geometry results for {config_name}: {e}\n")
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Successful: {len(successful)}/{len(configs_to_run)}")
    if successful:
        print(f"    {', '.join(successful)}")
    if failed:
        print(f"  Failed: {len(failed)}/{len(configs_to_run)}")
        for config_name, error in failed:
            print(f"    {config_name}: {error}")


if __name__ == "__main__":
    main()
