"""Entry point: ``python -m toy_task`` runs a tiny smoke experiment."""

from __future__ import annotations

from .config import RunConfig
from .training import run_experiment


def main() -> None:
    cfg = RunConfig(
        arch="dense", hidden_size=24, gamma=1.0, similarity=0.0, seed=0,
        epochs_per_phase=10,
    )
    res = run_experiment(cfg)
    print("toy_task smoke run:", cfg.run_id())
    print("  A1 end loss:", round(res.learning_curves["A1"][-1], 5))
    print("  forward transfer (T(s)):", round(res.behavioral["forward_transfer_Ts"], 5))
    print("  interference (T(0)):", round(res.behavioral["interference_T0"], 5))
    print("  extractions:", len(res.extractions))


if __name__ == "__main__":
    main()
