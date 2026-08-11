"""§B.5 ground-truth recovery sweeps for `src/glue/core.py`.

Acceptance criterion for the GLUE core (`02` §1, `05` §3): as each ground-truth
knob is swept with the others fixed, the corresponding effective measure must move
monotonically and in the sign the theory requires. Chou et al. App. B.5 settings:
P = 2, M = 200, N = 1000, D_ground = 4, R_ground = 1 outside the swept axis.

Writes `results/glue_core_recovery.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.glue import core
from src.manifolds import generator

P, M, N = 2, 200, 1000
D_FIXED, R_FIXED = 4, 1.0
N_T = 200
SEEDS = (0, 1, 2)

D_SWEEP = (2, 4, 6, 8, 10)
R_SWEEP = (0.8, 1.0, 1.4, 1.7, 2.0)
RHO_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8)


def measure(*, D: int, R: float, rho_C: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    arr = generator.make_arrangement(P, N, D, R, M, rng, rho_C=rho_C)
    res = core.glue_measures(
        core.from_arrangement(arr.points), np.random.default_rng(seed + 1000), n_t=N_T
    )
    return res.to_dict()


def sweep(name: str, key: str, values) -> list[dict]:
    rows = []
    for v in values:
        kw = dict(D=D_FIXED, R=R_FIXED, rho_C=0.0)
        kw[key] = v
        reps = [measure(seed=s, **kw) for s in SEEDS]
        rows.append(
            {
                "ground_truth": {key: v},
                **{
                    m: {
                        "mean": float(np.mean([r[m] for r in reps])),
                        "sd": float(np.std([r[m] for r in reps], ddof=1)),
                    }
                    for m in (
                        "alpha",
                        "D_eff",
                        "R_eff",
                        "Psi_eff",
                        "rho_c_glue",
                        "rho_c_signed",
                        "active_fraction",
                        "identity_residual",
                    )
                },
            }
        )
        last = rows[-1]
        print(
            f"  {key}={v:<5} alpha={last['alpha']['mean']:.4f} "
            f"D_eff={last['D_eff']['mean']:.4f} R_eff={last['R_eff']['mean']:.4f} "
            f"Psi={last['Psi_eff']['mean']:.4f} rho_c={last['rho_c_glue']['mean']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    t0 = time.time()
    out = {
        "estimator": "glue_core",
        "settings": {
            "P": P, "M": M, "N": N, "n_t": N_T,
            "D_fixed": D_FIXED, "R_fixed": R_FIXED, "seeds": list(SEEDS),
            "inactive_policy": "zero", "center_policy": "all", "kappa": 0,
        },
        "source": "Chou et al. ICML 2025 App. B.5 recovery settings",
    }
    for name, key, values in (
        ("dimension", "D", D_SWEEP),
        ("radius", "R", R_SWEEP),
        ("center_correlation", "rho_C", RHO_SWEEP),
    ):
        print(f"[{name}]", flush=True)
        out[name] = sweep(name, key, values)
    out["elapsed_s"] = round(time.time() - t0, 1)

    path = Path(__file__).resolve().parents[1] / "results" / "glue_core_recovery.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
