# Compute functional specialization metrics (retraining, correlation, ablations) for a run.
# Usage (from a1b2_modular or project root with --base-folder):
#   python scripts/03_functional_specialization.py <run_id>
#   python scripts/03_functional_specialization.py two_module_rnn_50 --no-ablations
#   python scripts/03_functional_specialization.py two_module_rnn_50 --participants study1_same_sub20

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_root = _script_dir.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from a1b2.analysis import run_loader
from a1b2.analysis import transfer_interference as ann
from a1b2.analysis.retraining_a1b2 import (
    REGRESSION_CHANCE,
    create_retraining_model_a1b2,
    train_probe_readout_a1b2,
    eval_probe_readout_a1b2,
    retraining_specialization_scalar,
    compute_ablations_metric_a1b2,
)
from a1b2.analysis.correlations_a1b2 import compute_correlation_metric_a1b2
from a1b2.data.basic_funcs import get_datasets
from a1b2.models.ffn import CreateParticipantDataset


def _project_root(base_folder):
    return os.path.abspath(os.path.normpath(base_folder))


def _jsonable(x):
    """Recursively convert numpy/torch types to JSON-safe native Python objects."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def _safe_mean_sem(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "sem": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    sem = float(np.std(arr, ddof=0) / (len(arr) ** 0.5)) if len(arr) > 0 else None
    return {"mean": float(np.mean(arr)), "sem": sem, "n": int(len(arr))}


def _parse_similarity(participant_id):
    p = str(participant_id)
    if "_same_" in p:
        return "same"
    if "_near_" in p:
        return "near"
    if "_far_" in p:
        return "far"
    return "unknown"


def _active_output_paths(sim_folder, output_tag=None):
    """Return output paths for active variant; tagged outputs when requested."""
    sim_folder = Path(sim_folder)
    canonical = {
        "json": sim_folder / "specialization_metrics.json",
        "participants_csv": sim_folder / "specialization_metrics_participants.csv",
        "participants_long_csv": sim_folder / "specialization_metrics_participants_long.csv",
        "summary_csv": sim_folder / "specialization_metrics_summary.csv",
        "audit_csv": sim_folder / "specialization_audit.csv",
    }
    if not output_tag:
        return canonical, None
    safe_tag = str(output_tag).replace(" ", "_")
    tagged = {
        "json": sim_folder / f"specialization_metrics__{safe_tag}.json",
        "participants_csv": sim_folder / f"specialization_metrics_participants__{safe_tag}.csv",
        "participants_long_csv": sim_folder / f"specialization_metrics_participants_long__{safe_tag}.csv",
        "summary_csv": sim_folder / f"specialization_metrics_summary__{safe_tag}.csv",
        "audit_csv": sim_folder / f"specialization_audit__{safe_tag}.csv",
    }
    return tagged, canonical


def _load_existing_results(active_json_path):
    if not os.path.isfile(active_json_path):
        return {}
    try:
        with open(active_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    existing = {}
    for row in data.get("participants", []):
        pid = row.get("id")
        if pid is not None:
            existing[str(pid)] = row
    return existing


def _participant_complete(existing_row, needs_retrain, needs_corr, needs_ablation):
    """Check whether cached row has enough fields to skip recomputation."""
    if not isinstance(existing_row, dict):
        return False
    if needs_retrain:
        det = existing_row.get("retraining_details")
        if not isinstance(det, dict) or "acc_matrix_3x2" not in det:
            return False
        if "retraining_specialization" not in existing_row:
            return False
    if needs_corr:
        det = existing_row.get("correlation_details")
        if not isinstance(det, dict):
            return False
        required = [
            "base_correlations",
            "correlations_fix_feature0",
            "correlations_fix_feature1",
            "norm_fix0",
            "norm_fix1",
        ]
        if det.get("status") == "ok":
            if any(k not in det for k in required):
                return False
        if "correlation_specialization" not in existing_row:
            return False
    if needs_ablation:
        det = existing_row.get("ablation_details")
        if not isinstance(det, dict) or "acc_matrix_3x2" not in det:
            return False
        if "ablation_specialization" not in existing_row:
            return False
    return True


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_loader_for_participant(df, participant, task_parameters, batch_size=32, shuffle=False):
    """Build a single DataLoader over A1 + B + A2 for one participant."""
    dataset_A1, dataset_B, dataset_A2, _, _ = get_datasets(df, participant, task_parameters)
    combined = ConcatDataset([
        CreateParticipantDataset(dataset_A1),
        CreateParticipantDataset(dataset_B),
        CreateParticipantDataset(dataset_A2),
    ])
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(
        description="Compute functional specialization metrics for a simulation run."
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run id (e.g. two_module_rnn_50) or path to sim folder.",
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        default="./",
        help="Base project folder (default: current dir).",
    )
    parser.add_argument(
        "--participants",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these participant ids. Default: all with state+npz.",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip retraining-specialization metric.",
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="Skip correlation-specialization metric.",
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip ablation-specialization metric.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for loaders (default: 32).",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Epochs for probe readout training (default: 5).",
    )
    parser.add_argument(
        "--regression-chance",
        type=float,
        default=REGRESSION_CHANCE,
        help="Expected accuracy of random predictor for chance correction (default: %.2f)." % REGRESSION_CHANCE,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all participants even if cached outputs exist.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Skip participants already fully present in output (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable cache-based skipping and recompute all participants unless --force is set.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional suffix for side-by-side output files (e.g. policy or date).",
    )
    args = parser.parse_args()

    root = _project_root(args.base_folder)
    data_folder = os.path.join(root, "data")
    # Allow run_id to be a path to the sim folder
    if os.path.isdir(args.run_id):
        sim_folder = os.path.abspath(args.run_id)
        run_id = os.path.basename(sim_folder.rstrip(os.sep))
    else:
        run_id = args.run_id
        sim_folder = os.path.join(data_folder, "simulations", run_id)

    if not os.path.isdir(sim_folder):
        raise SystemExit(f"Sim folder not found: {sim_folder}")

    settings = run_loader.load_settings(sim_folder)
    task_parameters = settings.get("task_parameters")
    if not task_parameters:
        task_parameters = ann.setup_task_parameters()

    df = ann.load_participant_data(data_folder)
    available_participants = set(df["participant"].unique())
    participants_with_npz = set(run_loader.list_participants_with_npz(sim_folder))
    participants_with_state = set(run_loader.list_participants_with_state(sim_folder))
    requested_participants = set(args.participants) if args.participants is not None else None

    participants = sorted(participants_with_npz & participants_with_state & available_participants)
    if requested_participants is not None:
        participants = [p for p in participants if p in requested_participants]
    if not participants:
        raise SystemExit(
            "No participants found with both state_*.pt and sim_*.npz and present in trial data."
        )

    active_paths, canonical_paths = _active_output_paths(sim_folder, args.output_tag)
    existing_rows = _load_existing_results(active_paths["json"]) if (args.resume and not args.force) else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    summary_rows = []
    long_rows = []
    audit_rows = []

    needs_retrain = not args.no_retrain
    needs_corr = not args.no_correlation
    needs_ablation = not args.no_ablations

    audit_population = sorted(participants_with_npz | participants_with_state | set(participants))
    if requested_participants is not None:
        audit_population = sorted(set(audit_population) | requested_participants)
    for p in audit_population:
        reasons = []
        if requested_participants is not None and p not in requested_participants:
            reasons.append("not_requested")
        if p not in participants_with_npz:
            reasons.append("missing_npz")
        if p not in participants_with_state:
            reasons.append("missing_state")
        if p not in available_participants:
            reasons.append("missing_trial_data")
        included = (len(reasons) == 0)
        if requested_participants is not None and p not in requested_participants:
            included = False
        audit_rows.append({
            "participant": p,
            "included": included,
            "reasons": ";".join(reasons) if reasons else "ok",
            "has_npz": p in participants_with_npz,
            "has_state": p in participants_with_state,
            "in_trial_data": p in available_participants,
            "requested": True if requested_participants is None else (p in requested_participants),
        })

    for participant in participants:
        if existing_rows and _participant_complete(
            existing_rows.get(participant), needs_retrain, needs_corr, needs_ablation
        ):
            row = existing_rows[participant]
            row.setdefault("diagnostics", {})
            row["diagnostics"]["cached"] = True
            results.append(row)
            continue

        row = {
            "id": participant,
            "similarity": _parse_similarity(participant),
            "diagnostics": {
                "cached": False,
                "state_exists": os.path.isfile(os.path.join(sim_folder, f"state_{participant}.pt")),
                "npz_exists": os.path.isfile(os.path.join(sim_folder, f"sim_{participant}.npz")),
                "warnings": [],
            },
        }
        if not args.no_retrain:
            row["retraining_specialization"] = None
        if not args.no_correlation:
            row["correlation_specialization"] = None
        if not args.no_ablations:
            row["ablation_specialization"] = None

        loader = build_loader_for_participant(
            df, participant, task_parameters,
            batch_size=args.batch_size, shuffle=True,
        )
        wrapper = run_loader.build_wrapper_from_settings(settings, device=device)
        state_path = os.path.join(sim_folder, f"state_{participant}.pt")
        run_loader.load_wrapper_state(wrapper, state_path)

        if not args.no_retrain:
            create_retraining_model_a1b2(wrapper, device=device)
            train_probe_readout_a1b2(
                wrapper, loader, settings.get("condition", {}),
                n_epochs=args.n_epochs, lr=1e-3, device=device,
            )
            acc = eval_probe_readout_a1b2(wrapper, loader, device)
            row["retraining_details"] = {
                "acc_matrix_3x2": _jsonable(acc),
                "acc_M0_f0": float(acc[0, 0]),
                "acc_M1_f0": float(acc[1, 0]),
                "acc_M0_f1": float(acc[0, 1]),
                "acc_M1_f1": float(acc[1, 1]),
            }
            row["retraining_specialization"] = float(retraining_specialization_scalar(
                acc[0, 0], acc[1, 0], acc[0, 1], acc[1, 1], chance=args.regression_chance
            ))

        if not args.no_correlation:
            npz_path = os.path.join(sim_folder, f"sim_{participant}.npz")
            with np.load(npz_path, allow_pickle=True) as data:
                if "hiddens_per_module" not in data or "probes" not in data or "inputs" not in data:
                    row["correlation_specialization"] = None
                    row["correlation_details"] = {
                        "status": "missing_required_keys",
                        "required": ["hiddens_per_module", "probes", "inputs"],
                        "present_keys": [k for k in ("hiddens_per_module", "probes", "inputs") if k in data],
                    }
                    row["diagnostics"]["warnings"].append("correlation_missing_required_keys")
                else:
                    participant_data = {
                        "hiddens_per_module": data["hiddens_per_module"],
                        "probes": data["probes"],
                        "inputs": data["inputs"],
                    }
                    out = compute_correlation_metric_a1b2(participant_data, n_samples=10)
                    row["correlation_details"] = {
                        "status": "ok",
                        "base_correlations": _jsonable(out["base_correlations"]),
                        "correlations_fix_feature0": _jsonable(out["correlations_fix_feature0"]),
                        "correlations_fix_feature1": _jsonable(out["correlations_fix_feature1"]),
                        "norm_fix0": _jsonable(out["norm_correlations_fix0"]),
                        "norm_fix1": _jsonable(out["norm_correlations_fix1"]),
                    }
                    row["correlation_specialization"] = float(out["correlation_specialization"])

        if not args.no_ablations:
            if args.no_retrain:
                # Need probe readout for ablations; build and train if we skipped retrain
                create_retraining_model_a1b2(wrapper, device=device)
                train_probe_readout_a1b2(
                    wrapper, loader, settings.get("condition", {}),
                    n_epochs=args.n_epochs, lr=1e-3, device=device,
                )
            ab = compute_ablations_metric_a1b2(wrapper, loader, device, chance=args.regression_chance)
            ab_acc = np.asarray(ab["acc"])
            row["ablation_details"] = {
                "acc_matrix_3x2": _jsonable(ab_acc),
                "acc_M0_f0": float(ab_acc[0, 0]),
                "acc_M1_f0": float(ab_acc[1, 0]),
                "acc_M0_f1": float(ab_acc[0, 1]),
                "acc_M1_f1": float(ab_acc[1, 1]),
            }
            row["ablation_specialization"] = float(ab["ablation_specialization"])

        row["diagnostics"]["warnings"] = _jsonable(row["diagnostics"]["warnings"])
        results.append(row)

    # Aggregate summary
    mean_sem = {}
    for key in ("retraining_specialization", "correlation_specialization", "ablation_specialization"):
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            mean_sem[key] = {"mean": float(np.mean(vals)), "sem": float(np.std(vals) / (len(vals) ** 0.5))}
        else:
            mean_sem[key] = {"mean": None, "sem": None}

    summary = {
        "retraining_specialization": _safe_mean_sem([r.get("retraining_specialization") for r in results]),
        "correlation_specialization": _safe_mean_sem([r.get("correlation_specialization") for r in results]),
        "ablation_specialization": _safe_mean_sem([r.get("ablation_specialization") for r in results]),
    }

    # Analysis-ready tables
    for r in results:
        summary_rows.append({
            "run_id": run_id,
            "participant": r.get("id"),
            "similarity": r.get("similarity", "unknown"),
            "retraining_specialization": r.get("retraining_specialization"),
            "correlation_specialization": r.get("correlation_specialization"),
            "ablation_specialization": r.get("ablation_specialization"),
            "cached": bool(r.get("diagnostics", {}).get("cached", False)),
            "warnings": ";".join(r.get("diagnostics", {}).get("warnings", [])) if isinstance(r.get("diagnostics", {}).get("warnings", []), list) else r.get("diagnostics", {}).get("warnings", ""),
        })
        for metric_key in ("retraining_specialization", "correlation_specialization", "ablation_specialization"):
            long_rows.append({
                "run_id": run_id,
                "participant": r.get("id"),
                "similarity": r.get("similarity", "unknown"),
                "metric": metric_key,
                "value": r.get(metric_key),
            })

    out = {
        # Legacy keys kept for non-breaking compatibility
        "run_id": run_id,
        "mean": mean_sem,
        "participants": _jsonable(results),
        # New structured blocks
        "run": {
            "run_id": run_id,
            "sim_folder": sim_folder,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "script": "scripts/03_functional_specialization.py",
            "args": _jsonable(vars(args)),
            "settings_condition": _jsonable(settings.get("condition", {})),
            "settings_training_params": _jsonable(settings.get("training_params", {})),
            "settings_task_parameters": _jsonable(task_parameters),
            "output_tag": args.output_tag,
        },
        "methodology": {
            "regression_chance": float(args.regression_chance),
            "batch_size": int(args.batch_size),
            "probe_epochs": int(args.n_epochs),
            "enabled_metrics": {
                "retraining": needs_retrain,
                "correlation": needs_corr,
                "ablations": needs_ablation,
            },
            "formula_guardrail": "No scalar formula changes; only persistence expanded.",
        },
        "summary": summary,
        "audit": {
            "participants_trial_data_count": int(len(available_participants)),
            "participants_npz_count": int(len(participants_with_npz)),
            "participants_state_count": int(len(participants_with_state)),
            "participants_included_count": int(len(participants)),
        },
    }

    # Write active outputs
    with open(active_paths["json"], "w") as f:
        json.dump(_jsonable(out), f, indent=2)

    _write_csv(
        active_paths["participants_csv"],
        summary_rows,
        [
            "run_id",
            "participant",
            "similarity",
            "retraining_specialization",
            "correlation_specialization",
            "ablation_specialization",
            "cached",
            "warnings",
        ],
    )
    _write_csv(
        active_paths["participants_long_csv"],
        long_rows,
        ["run_id", "participant", "similarity", "metric", "value"],
    )
    _write_csv(
        active_paths["summary_csv"],
        [
            {
                "run_id": run_id,
                "metric": k,
                "mean": v["mean"],
                "sem": v["sem"],
                "n": v["n"],
            }
            for k, v in summary.items()
        ],
        ["run_id", "metric", "mean", "sem", "n"],
    )
    _write_csv(
        active_paths["audit_csv"],
        audit_rows,
        ["participant", "included", "reasons", "has_npz", "has_state", "in_trial_data", "requested"],
    )

    # Keep canonical outputs for backward compatibility when using output tags.
    if canonical_paths is not None:
        with open(canonical_paths["json"], "w") as f:
            json.dump(_jsonable(out), f, indent=2)
        _write_csv(canonical_paths["participants_csv"], summary_rows, ["run_id", "participant", "similarity", "retraining_specialization", "correlation_specialization", "ablation_specialization", "cached", "warnings"])
        _write_csv(canonical_paths["participants_long_csv"], long_rows, ["run_id", "participant", "similarity", "metric", "value"])
        _write_csv(canonical_paths["summary_csv"], [{"run_id": run_id, "metric": k, "mean": v["mean"], "sem": v["sem"], "n": v["n"]} for k, v in summary.items()], ["run_id", "metric", "mean", "sem", "n"])
        _write_csv(canonical_paths["audit_csv"], audit_rows, ["participant", "included", "reasons", "has_npz", "has_state", "in_trial_data", "requested"])

    print(f"Wrote {active_paths['json']}")
    print(f"Wrote {active_paths['participants_csv']}")
    print(f"Wrote {active_paths['participants_long_csv']}")
    print(f"Wrote {active_paths['summary_csv']}")
    print(f"Wrote {active_paths['audit_csv']}")
    if canonical_paths is not None:
        print(f"Also updated canonical outputs in {sim_folder}")


if __name__ == "__main__":
    main()
