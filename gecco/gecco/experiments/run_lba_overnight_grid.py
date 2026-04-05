"""
Terminal entry point for the LBA overnight experiment grid (same logic as notebooks/lba_overnight_grid.ipynb).

Run from the gecco install (after ``pip install -e .``), typically with cwd = repo ``gecco/``:

    CUDA_VISIBLE_DEVICES=4,5,6,7 GECCO_CUDA_DEVICE=0 \\
      python -m gecco.experiments.run_lba_overnight_grid --grid-mode extensive

Or pass GPU flags (applied before importing torch):

    python -m gecco.experiments.run_lba_overnight_grid \\
      --cuda-visible-devices 4,5,6,7 --cuda-device 0 --grid-mode extensive
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# CLI first — set CUDA before torch
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LBA overnight NSGA-II grid → CSV (resume-safe).")
    p.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="e.g. 4,5,6,7. Only used if CUDA_VISIBLE_DEVICES is not already set.",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Logical GPU index inside the visible set (default 0 or GECCO_CUDA_DEVICE).",
    )
    p.add_argument(
        "--grid-mode",
        choices=("smoke", "standard", "extensive", "mega"),
        default="extensive",
    )
    p.add_argument("--trial-df", type=str, default=None, help="Override GECCO_TRIAL_DF / default path.")
    p.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Parent directory for timestamped runs (default: <gecco>/runs/overnight_nb).",
    )
    p.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Existing run folder with manifest.csv (append new rows; same as GECCO_OVERNIGHT_RESUME).",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    p.add_argument(
        "--init-scales",
        type=str,
        default=None,
        help="Comma-separated σ list for extensive/mega/standard (overrides built-in list for that mode).",
    )
    return p


def _bootstrap_cuda_env() -> None:
    """Parse only CUDA-related flags so CUDA_VISIBLE_DEVICES is set before torch import."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--cuda-device", type=int, default=None)
    early, _ = p.parse_known_args()
    if early.cuda_visible_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = early.cuda_visible_devices
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GECCO_CUDA_VISIBLE_DEVICES", "4,5,6,7")
    if early.cuda_device is not None:
        os.environ["GECCO_CUDA_DEVICE"] = str(early.cuda_device)


def _parse_init_scales(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    _bootstrap_cuda_env()

import pandas as pd
import torch

from gecco.data.holton_trials import load_trial_df, pick_participants
from gecco.evolve.nsga2 import run_nsga2
from gecco.experiments.run_lba_figure import aggregate_cyclic_tensors, pareto_indices, _default_trial_df
from gecco.models.routing_rnn import DualRouteRNN, SingleRouteRNN, routing_separation_score
from gecco.training.episode import apply_init_scale, train_on_schedule

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


# Default σ sweeps (match notebook)
INIT_SCALES_STANDARD = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0]
INIT_SCALES_EXTENSIVE = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
INIT_SCALES_MEGA = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


@dataclass(frozen=True)
class ExperimentSpec:
    condition: str
    seed: int
    init_scale: float
    cyclic_pattern: str
    participants_per_condition: int

    def run_id(self) -> str:
        pat = self.cyclic_pattern.replace("/", "")
        return f"c{self.condition}_s{self.seed}_i{self.init_scale}_p{pat}_ppc{self.participants_per_condition}"


def build_experiments(
    mode: str,
    *,
    init_scales_standard: List[float],
    init_scales_extensive: List[float],
    init_scales_mega: List[float],
) -> List[ExperimentSpec]:
    if mode == "smoke":
        return [ExperimentSpec("near", 0, 0.1, "ABABAB", 2)]
    if mode == "standard":
        conds = ["near", "far"]
        seeds = [0, 1]
        inits = list(init_scales_standard)
        pats = ["ABABAB"]
        ppc = [2, 3]
    elif mode == "extensive":
        conds = ["near", "far", "same"]
        seeds = [0, 1, 2, 3]
        inits = list(init_scales_extensive)
        pats = ["ABABAB", "ABABABAB"]
        ppc = [2, 3]
    elif mode == "mega":
        conds = ["near", "far", "same"]
        seeds = [0, 1, 2, 3, 4]
        inits = list(init_scales_mega)
        pats = ["ABABAB", "ABABABAB"]
        ppc = [2, 3]
    else:
        raise ValueError(mode)

    out: List[ExperimentSpec] = []
    for c, s, i, pat, p in product(conds, seeds, inits, pats, ppc):
        out.append(ExperimentSpec(str(c), int(s), float(i), str(pat), int(p)))
    return out


def _gecco_project_root() -> Path:
    # gecco/gecco/experiments/this_file.py -> parents[2] == gecco project (pyproject.toml)
    return Path(__file__).resolve().parents[2]


def _ensure_csv(path: Path, fieldnames: List[str]) -> None:
    if path.is_file():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def load_completed_run_ids(manifest: Path) -> set:
    if not manifest.is_file():
        return set()
    df = pd.read_csv(manifest)
    ok = df.loc[df.get("status", "") == "ok", "run_id"]
    return set(ok.astype(str).tolist())


MANIFEST_FIELDS = [
    "run_id",
    "status",
    "error",
    "duration_s",
    "utc_finished",
    "condition",
    "seed",
    "init_scale",
    "cyclic_pattern",
    "participants_per_condition",
    "population",
    "generations",
    "train_steps",
    "participants_json",
    "pareto_n",
    "pop_n",
    "best_mse_pareto",
    "worst_mse_pareto",
    "max_routing_sep_pareto",
    "baseline_low_sigma_mse",
    "baseline_high_sigma_mse",
]
PARETO_FIELDS = [
    "run_id",
    "condition",
    "seed",
    "init_scale",
    "cyclic_pattern",
    "participants_per_condition",
    "mse",
    "f2_neg_sep",
    "routing_separation",
    "genome",
]
SAMPLE_FIELDS = ["run_id", "condition", "mse", "routing_separation", "genome"]


def append_rows(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            w.writerow(r)
        f.flush()


def run_one(
    ex: ExperimentSpec,
    cfg: dict,
    df,
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pids = pick_participants(
        df,
        conditions=[ex.condition],
        per_condition=ex.participants_per_condition,
        seed=ex.seed,
    )
    if len(pids) < ex.participants_per_condition:
        raise RuntimeError(f"only {len(pids)} participants for {ex.condition}: {pids}")

    tensors = aggregate_cyclic_tensors(
        df.loc[df["participant"].isin(pids)],
        pids,
        max_trials_per_segment=24,
        pattern=ex.cyclic_pattern,
        seed=ex.seed,
    )

    def make_evaluate(tensors_d: Dict[str, torch.Tensor], c: dict):
        def evaluate(genome: List[int]) -> Tuple[float, float]:
            m = DualRouteRNN(genome, hidden_per_module=c["hidden"])
            apply_init_scale(m, c["init_scale"])
            _, val = train_on_schedule(
                m,
                tensors_d,
                steps=c["train_steps"],
                lr=c["lr"],
                batch_size=c["batch_size"],
                device=device,
                seed=c["seed"] + (hash(tuple(genome)) % 997),
            )
            sep = routing_separation_score(genome)
            return val, -sep

        return evaluate

    def run_baselines(tensors_d: Dict[str, torch.Tensor], c: dict) -> List[Tuple[str, float]]:
        rows_b = []
        for sigma, tag in [(c["baseline_init_low"], "low_sigma"), (c["baseline_init_high"], "high_sigma")]:
            sm = SingleRouteRNN(hidden_per_module=c["hidden"])
            apply_init_scale(sm, sigma)
            _, val = train_on_schedule(
                sm,
                tensors_d,
                steps=c["train_steps"] * 2,
                lr=c["lr"],
                batch_size=c["batch_size"],
                device=device,
                seed=c["seed"],
            )
            rows_b.append((tag, float(val)))
        return rows_b

    evaluate = make_evaluate(tensors, cfg)
    pop = run_nsga2(
        evaluate,
        genome_length=12,
        population_size=cfg["population"],
        generations=cfg["generations"],
        seed=cfg["seed"] + hash(ex.condition) % 10000,
    )
    front = pareto_indices(pop)
    baselines = run_baselines(tensors, cfg)

    rid = ex.run_id()
    mses = [ind.f1 for ind in front]
    seps = [routing_separation_score(ind.genome) for ind in front]
    bl = dict(baselines)

    manifest_row = {
        "run_id": rid,
        "status": "ok",
        "error": "",
        "duration_s": "",
        "utc_finished": datetime.now(timezone.utc).isoformat(),
        "condition": ex.condition,
        "seed": ex.seed,
        "init_scale": ex.init_scale,
        "cyclic_pattern": ex.cyclic_pattern,
        "participants_per_condition": ex.participants_per_condition,
        "population": cfg["population"],
        "generations": cfg["generations"],
        "train_steps": cfg["train_steps"],
        "participants_json": json.dumps(pids),
        "pareto_n": len(front),
        "pop_n": len(pop),
        "best_mse_pareto": min(mses) if mses else "",
        "worst_mse_pareto": max(mses) if mses else "",
        "max_routing_sep_pareto": max(seps) if seps else "",
        "baseline_low_sigma_mse": bl.get("low_sigma", ""),
        "baseline_high_sigma_mse": bl.get("high_sigma", ""),
    }

    pareto_rows = []
    for ind in front:
        g = ind.genome
        pareto_rows.append(
            {
                "run_id": rid,
                "condition": ex.condition,
                "seed": ex.seed,
                "init_scale": ex.init_scale,
                "cyclic_pattern": ex.cyclic_pattern,
                "participants_per_condition": ex.participants_per_condition,
                "mse": ind.f1,
                "f2_neg_sep": ind.f2,
                "routing_separation": routing_separation_score(g),
                "genome": "".join(str(x) for x in g),
            }
        )

    sample_rows = []
    for ind in pop[:80]:
        g = ind.genome
        sample_rows.append(
            {
                "run_id": rid,
                "condition": ex.condition,
                "mse": ind.f1,
                "routing_separation": routing_separation_score(g),
                "genome": "".join(str(x) for x in g),
            }
        )

    return manifest_row, pareto_rows, sample_rows


def main() -> int:
    args = _build_parser().parse_args()
    cuda_idx = int(os.environ.get("GECCO_CUDA_DEVICE", "0"))

    trial_path = Path(args.trial_df or os.environ.get("GECCO_TRIAL_DF") or str(_default_trial_df())).expanduser()
    if not trial_path.is_file():
        print(f"ERROR: trial_df not found: {trial_path}", file=sys.stderr)
        return 1

    gecco_root = _gecco_project_root()
    run_root = Path(args.run_root or os.environ.get("GECCO_OVERNIGHT_RUN_DIR", "")).expanduser()
    if not str(run_root):
        run_root = gecco_root / "runs" / "overnight_nb"

    resume = (args.resume_run_dir or os.environ.get("GECCO_OVERNIGHT_RESUME", "")).strip()
    if resume:
        run_dir = Path(resume).expanduser().resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
        run_dir = (run_root / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = run_dir / "manifest.csv"
    pareto_csv = run_dir / "pareto_points.csv"
    pop_sample_csv = run_dir / "population_sample.csv"

    base = {
        "population": 44,
        "generations": 24,
        "train_steps": 180,
        "batch_size": 32,
        "lr": 0.02,
        "hidden": 32,
        "baseline_init_low": 0.001,
        "baseline_init_high": 2.0,
        "use_cpu": args.cpu,
    }
    smoke_ga = {"population": 12, "generations": 4, "train_steps": 40}

    iss = INIT_SCALES_STANDARD
    ise = INIT_SCALES_EXTENSIVE
    ism = INIT_SCALES_MEGA
    if args.init_scales:
        custom = _parse_init_scales(args.init_scales)
        iss = ise = ism = custom

    experiments = build_experiments(
        args.grid_mode,
        init_scales_standard=iss,
        init_scales_extensive=ise,
        init_scales_mega=ism,
    )

    print(f"grid_mode={args.grid_mode!r} -> {len(experiments)} jobs")
    print(f"  init_scale values: {sorted({e.init_scale for e in experiments})}")
    print(f"trial_df: {trial_path}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

    if base["use_cpu"] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        n = torch.cuda.device_count()
        idx = min(max(cuda_idx, 0), n - 1) if n else 0
        device = torch.device(f"cuda:{idx}")
    print(f"device: {device}")
    print(f"RUN_DIR: {run_dir}")

    grid_path = run_dir / "grid.jsonl"
    if not grid_path.is_file():
        with grid_path.open("w", encoding="utf-8") as f:
            for ex in experiments:
                f.write(json.dumps(ex.__dict__, sort_keys=True) + "\n")
        print("wrote", grid_path)
    else:
        print("keep existing", grid_path)

    _ensure_csv(manifest_csv, MANIFEST_FIELDS)
    _ensure_csv(pareto_csv, PARETO_FIELDS)
    _ensure_csv(pop_sample_csv, SAMPLE_FIELDS)

    df = load_trial_df(str(trial_path))
    cfg = {**base, "init_scale": 0.1}
    if args.grid_mode == "smoke":
        cfg = {**cfg, **smoke_ga}

    done = load_completed_run_ids(manifest_csv)
    print("already completed (ok):", len(done))

    errors: List[Tuple[str, str]] = []
    for ex in tqdm(experiments, desc="overnight grid"):
        rid = ex.run_id()
        if rid in done:
            continue
        row_cfg = {**cfg, "seed": ex.seed, "init_scale": ex.init_scale}
        t0 = time.perf_counter()
        try:
            manifest_row, pareto_rows, sample_rows = run_one(ex, row_cfg, df, device)
            dt = time.perf_counter() - t0
            manifest_row["duration_s"] = f"{dt:.3f}"
            manifest_row["utc_finished"] = datetime.now(timezone.utc).isoformat()
            append_rows(manifest_csv, MANIFEST_FIELDS, [manifest_row])
            append_rows(pareto_csv, PARETO_FIELDS, pareto_rows)
            append_rows(pop_sample_csv, SAMPLE_FIELDS, sample_rows)
            print(f"OK {rid} ({dt:.1f}s)")
        except Exception as e:
            dt = time.perf_counter() - t0
            err = traceback.format_exc()
            append_rows(
                manifest_csv,
                MANIFEST_FIELDS,
                [
                    {
                        "run_id": rid,
                        "status": "error",
                        "error": repr(e),
                        "duration_s": f"{dt:.3f}",
                        "utc_finished": datetime.now(timezone.utc).isoformat(),
                        "condition": ex.condition,
                        "seed": ex.seed,
                        "init_scale": ex.init_scale,
                        "cyclic_pattern": ex.cyclic_pattern,
                        "participants_per_condition": ex.participants_per_condition,
                        "population": cfg["population"],
                        "generations": cfg["generations"],
                        "train_steps": cfg["train_steps"],
                        "participants_json": "",
                        "pareto_n": "",
                        "pop_n": "",
                        "best_mse_pareto": "",
                        "worst_mse_pareto": "",
                        "max_routing_sep_pareto": "",
                        "baseline_low_sigma_mse": "",
                        "baseline_high_sigma_mse": "",
                    }
                ],
            )
            with (run_dir / "tracebacks.txt").open("a", encoding="utf-8") as f:
                f.write(f"\n===== {rid} =====\n{err}\n")
            errors.append((rid, repr(e)))
            print(f"ERR {rid}: {e}")

    print("done. errors:", len(errors))
    if errors:
        print(errors[:5], "...")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
