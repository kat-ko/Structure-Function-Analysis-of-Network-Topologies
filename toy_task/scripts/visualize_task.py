"""Stage-1 visualization (Part V Section 2): the environment, no neural network.

Renders, for one seed:
  - latent objects Z and clean observations X (PCA to 2-D),
  - target rings for a range of similarity values s,
demonstrating that only the targets change with s while Z and X stay fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from toy_task.config import SIMILARITY_GRID, CONSTANTS
from toy_task.environment import Environment


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize the toy_task environment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="figures/task_overview.png")
    args = p.parse_args()

    env = Environment.from_seed(args.seed)
    Z = env.Z
    X = env.clean_observation()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (1) latent objects in 2-D PCA
    z2 = PCA(n_components=2).fit_transform(Z)
    axes[0].scatter(z2[:, 0], z2[:, 1], c=range(CONSTANTS.n_objects), cmap="hsv", s=120)
    for i, (a, b) in enumerate(z2):
        axes[0].annotate(str(i), (a, b), fontsize=9)
    axes[0].set_title(f"Latent objects Z (PCA)  seed={args.seed}")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # (2) clean observations in 2-D PCA
    x2 = PCA(n_components=2).fit_transform(X)
    axes[1].scatter(x2[:, 0], x2[:, 1], c=range(CONSTANTS.n_objects), cmap="hsv", s=120)
    for i, (a, b) in enumerate(x2):
        axes[1].annotate(str(i), (a, b), fontsize=9)
    axes[1].set_title("Clean observations X = tanh(Wz) (PCA)")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    # (3) target rings for several s (rigid rotation)
    show_s = [0.0, np.pi / 4, np.pi / 2, np.pi]
    for s in show_s:
        Y = env.targets(s)
        axes[2].scatter(Y[:, 0], Y[:, 1], s=60, label=f"s={s:.2f}")
    axes[2].set_aspect("equal")
    axes[2].set_title("Target rings T(s): rigid rotation by s")
    axes[2].set_xlabel("cos"); axes[2].set_ylabel("sin")
    axes[2].legend(fontsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # Console invariance checks (Stage-1 gate).
    X2 = env.clean_observation()
    assert np.allclose(X, X2), "clean observations should be deterministic"
    for s in SIMILARITY_GRID:
        Y = env.targets(s)
        norms = np.linalg.norm(Y, axis=1)
        assert np.allclose(norms, 1.0), f"targets must lie on unit circle (s={s})"
    print("invariance checks passed: X deterministic, all targets on unit circle")


if __name__ == "__main__":
    main()
