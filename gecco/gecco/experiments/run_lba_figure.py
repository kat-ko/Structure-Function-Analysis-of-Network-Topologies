"""
GECCO LBA default figure: NSGA-II on input-routing genomes vs single-module baseline.

Run from repo with trial_df available, after ``pip install -e gecco/``:

    python -m gecco.experiments.run_lba_figure --trial-df path/to/trial_df.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from gecco.data.holton_trials import (
    build_cyclic_ab_schedule,
    get_datasets,
    load_trial_df,
    pick_participants,
    setup_task_parameters,
)
from gecco.evolve.nsga2 import Individual, _dominates, run_nsga2
from gecco.models.routing_rnn import DualRouteRNN, SingleRouteRNN, routing_separation_score
from gecco.training.episode import apply_init_scale, train_on_schedule


def _repo_root() -> Path:
    # gecco/gecco/experiments/run_lba_figure.py -> parents[3] = monorepo root
    return Path(__file__).resolve().parents[3]


def _default_trial_df() -> Path:
    env = (os.environ.get("GECCO_TRIAL_DF") or "").strip()
    # Ignore bogus env (e.g. GECCO_TRIAL_DF=. from a mistaken export) so we fall back to monorepo path.
    if env and env not in (".", "..") and not env.endswith("/."):
        return Path(env).expanduser()
    return _repo_root() / "transfer-interference" / "data" / "participants" / "trial_df.csv"


def aggregate_cyclic_tensors(
    df,
    participants: List[str],
    *,
    max_trials_per_segment: Optional[int],
    pattern: str,
    seed: int,
):
    """Concatenate cyclic schedules across participants (same condition slice)."""
    task_params = setup_task_parameters()
    chunks = []
    for i, pid in enumerate(participants):
        A1, B, A2, _, _ = get_datasets(df, pid, task_params)
        _ = A2  # cyclic ABABAB uses A1 + B only in LBA simplification
        t = build_cyclic_ab_schedule(
            A1,
            B,
            pattern=pattern,
            max_trials_per_segment=max_trials_per_segment,
            seed=seed + i * 17,
        )
        chunks.append(t)
    out = {}
    for k in chunks[0]:
        out[k] = torch.cat([c[k] for c in chunks], dim=0)
    return out


def pareto_indices(pop: List[Individual]) -> List[Individual]:
    nd: List[Individual] = []
    for p in pop:
        if not any(_dominates(q, p) for q in pop if q is not p):
            nd.append(p)
    return nd


def run_lba_figure(args: argparse.Namespace) -> None:
    trial_path = Path(args.trial_df).expanduser().resolve()
    if not trial_path.is_file():
        env = os.environ.get("GECCO_TRIAL_DF")
        hint = ""
        if env:
            hint = (
                f" Check GECCO_TRIAL_DF (currently {env!r}) — use a full path to trial_df.csv, "
                "or unset it to use the default under transfer-interference/."
            )
        raise FileNotFoundError(
            f"trial_df not found: {trial_path}.{hint} "
            f"Example: --trial-df $PWD/../transfer-interference/data/participants/trial_df.csv"
        )

    df = load_trial_df(str(trial_path))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    colors = {"near": "#1f77b4", "far": "#ff7f0e"}
    markers = {"near": "o", "far": "s"}

    for condition in args.conditions:
        pids = pick_participants(df, conditions=[condition], per_condition=args.participants_per_condition, seed=args.seed)
        if len(pids) < args.participants_per_condition:
            print(f"Warning: only {len(pids)} pids for {condition}: {pids}")

        tensors = aggregate_cyclic_tensors(
            df.loc[df["participant"].isin(pids)],
            pids,
            max_trials_per_segment=args.max_trials_per_segment,
            pattern=args.cyclic_pattern,
            seed=args.seed,
        )

        def evaluate(genome: List[int]) -> Tuple[float, float]:
            m = DualRouteRNN(genome, hidden_per_module=args.hidden)
            apply_init_scale(m, args.init_scale)
            _, val = train_on_schedule(
                m,
                tensors,
                steps=args.train_steps,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + (hash(tuple(genome)) % 997),
            )
            sep = routing_separation_score(genome)
            return val, -sep

        pop = run_nsga2(
            evaluate,
            genome_length=12,
            population_size=args.population,
            generations=args.generations,
            seed=args.seed + hash(condition) % 10000,
        )
        front = pareto_indices(pop)
        xs = [routing_separation_score(ind.genome) for ind in front]
        ys = [ind.f1 for ind in front]
        ax.scatter(
            xs,
            ys,
            c=colors.get(condition, "gray"),
            marker=markers.get(condition, "o"),
            alpha=0.85,
            s=55,
            label=f"evolved (Pareto) — {condition}",
            edgecolors="white",
            linewidths=0.5,
        )

        # Optional: faint cloud of final population
        if args.plot_all_population:
            ax.scatter(
                [routing_separation_score(i.genome) for i in pop],
                [i.f1 for i in pop],
                c=colors.get(condition, "gray"),
                alpha=0.12,
                s=18,
            )

    # Single-module baselines (no routing evolution; x=0 on routing axis)
    baseline_rows: List[Tuple[str, float, float, str]] = []
    for condition in args.conditions:
        pids = pick_participants(df, conditions=[condition], per_condition=args.participants_per_condition, seed=args.seed)
        tensors = aggregate_cyclic_tensors(
            df.loc[df["participant"].isin(pids)],
            pids,
            max_trials_per_segment=args.max_trials_per_segment,
            pattern=args.cyclic_pattern,
            seed=args.seed,
        )
        for sigma, tag in [(args.baseline_init_low, "low σ"), (args.baseline_init_high, "high σ")]:
            sm = SingleRouteRNN(hidden_per_module=args.hidden)
            apply_init_scale(sm, sigma)
            _, val = train_on_schedule(
                sm,
                tensors,
                steps=args.train_steps * 2,
                lr=args.lr,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
            )
            baseline_rows.append((condition, 0.0, val, tag))
    for condition, x, y, tag in baseline_rows:
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=220,
            c=colors.get(condition, "gray"),
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
            label=f"{condition} single ({tag})",
        )

    ax.set_xlabel("Task-split routing |mean(A slots) − mean(B slots)|")
    ax.set_ylabel("Validation MSE (probed cos/sin)")
    ax.set_title("Holton-style cyclic schedule — routing Pareto vs single-module")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trial-df",
        type=str,
        default=None,
        help="Path to trial_df.csv (default: GECCO_TRIAL_DF if set, else monorepo transfer-interference/.../trial_df.csv)",
    )
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[2] / "figures" / "lba_pareto.png"))
    p.add_argument("--conditions", nargs="+", default=["near", "far"])
    p.add_argument("--participants-per-condition", type=int, default=2)
    p.add_argument("--max-trials-per-segment", type=int, default=24)
    p.add_argument("--cyclic-pattern", type=str, default="ABABAB")
    p.add_argument("--population", type=int, default=28)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--init-scale", type=float, default=0.1, help="Global σ for evolved dual-module nets")
    p.add_argument("--baseline-init-low", type=float, default=0.001)
    p.add_argument("--baseline-init-high", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--plot-all-population", action="store_true")
    args = p.parse_args()
    if args.trial_df is None:
        args.trial_df = str(_default_trial_df())
    run_lba_figure(args)


if __name__ == "__main__":
    main()
