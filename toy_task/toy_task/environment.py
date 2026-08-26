"""Stimulus manifold, observation function and task family (Part II + patch Section 2-4).

A single seed fully determines one environment: latent identities ``Z``, the fixed
observation mapping ``W`` (``b=0``), and the per-epoch sampling order. Targets are a
rigid rotation of a per-object reference ring by the similarity parameter ``s``.

The per-object reference angle is a **function of the input** (the ``atan2`` of the
first two latent coordinates), so the task carries an abstract input->angle rule that
can generalize to unseen objects. This is what makes the ``novel`` stimulus regime a
genuine Holton-style rule-transfer test rather than fresh memorization. The ``novel``
regime gives task B its own independent object set ``Z_b``; the ``shared`` regime
reuses task A's objects for B (``Z_b is Z``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import (
    ANGLE_EVEN,
    ANGLE_MODES,
    ANGLE_RANDOM,
    CONSTANTS,
    STIMULUS_REGIMES,
    STIMULUS_SHARED,
    STIMULUS_NOVEL,
)


@dataclass
class Environment:
    """One independent realization of the benchmark environment.

    Attributes
    ----------
    seed : int
        Random seed defining this environment.
    Z : np.ndarray, shape (N, d_z)
        Task-A latent object identities, ``z_i ~ N(0, I)``.
    Z_b : np.ndarray, shape (N, d_z)
        Task-B latent object identities. Equal to ``Z`` in the ``shared`` regime;
        an independent fresh draw in the ``novel`` regime.
    W : np.ndarray, shape (d_x, d_z)
        Fixed observation projection, ``W_ij ~ N(0, 1/d_z)``.
    b : np.ndarray, shape (d_x,)
        Observation bias (zeros; kept for clarity / future use).
    sigma_train : float
        Std of additive Gaussian observation noise during training.
    stimulus_regime : str
        ``"shared"`` or ``"novel"`` (see ``config.STIMULUS_REGIMES``).
    angle_mode : str
        ``"random"`` or ``"even"`` (see ``config.ANGLE_MODES``).
    """

    seed: int
    Z: np.ndarray
    Z_b: np.ndarray
    W: np.ndarray
    b: np.ndarray
    sigma_train: float
    stimulus_regime: str = STIMULUS_SHARED
    angle_mode: str = ANGLE_EVEN

    # ------------------------------------------------------------------ build
    @staticmethod
    def _even_thetas(n_objects: int, phase_offset: float = 0.0) -> np.ndarray:
        """Patch-faithful equally spaced angles on the circle."""
        return np.linspace(0.0, 2.0 * np.pi, n_objects, endpoint=False) + float(phase_offset)

    @staticmethod
    def _embed_ring_angles(Z: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Set ``(z_{i,0}, z_{i,1}) = (cos theta_i, sin theta_i)``; other dims unchanged."""
        out = Z.copy()
        out[:, 0] = np.cos(thetas)
        out[:, 1] = np.sin(thetas)
        return out

    @classmethod
    def from_seed(
        cls,
        seed: int,
        sigma_train: float | None = None,
        stimulus_regime: str = STIMULUS_SHARED,
        angle_mode: str = ANGLE_EVEN,
    ) -> "Environment":
        if stimulus_regime not in STIMULUS_REGIMES:
            raise ValueError(
                f"Unknown stimulus_regime {stimulus_regime!r}; expected one of {STIMULUS_REGIMES}"
            )
        if angle_mode not in ANGLE_MODES:
            raise ValueError(
                f"Unknown angle_mode {angle_mode!r}; expected one of {ANGLE_MODES}"
            )
        c = CONSTANTS
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((c.n_objects, c.d_latent)).astype(np.float64)
        # W_ij ~ N(0, 1/d_z)  =>  std = sqrt(1/d_z)
        W = (rng.standard_normal((c.d_obs, c.d_latent)) / np.sqrt(c.d_latent)).astype(np.float64)
        b = np.zeros(c.d_obs, dtype=np.float64)
        if angle_mode == ANGLE_EVEN:
            Z = cls._embed_ring_angles(Z, cls._even_thetas(c.n_objects))
        if stimulus_regime == STIMULUS_NOVEL:
            # Fresh, independent object set for task B (drawn after Z and W so the
            # task-A environment is bit-identical across regimes for a given seed).
            Z_b = rng.standard_normal((c.n_objects, c.d_latent)).astype(np.float64)
            if angle_mode == ANGLE_EVEN:
                # Distinct but equally spaced B ring (half-step offset) for separability.
                Z_b = cls._embed_ring_angles(
                    Z_b, cls._even_thetas(c.n_objects, phase_offset=np.pi / c.n_objects)
                )
        else:
            Z_b = Z
        return cls(
            seed=seed,
            Z=Z,
            Z_b=Z_b,
            W=W,
            b=b,
            sigma_train=c.sigma_train if sigma_train is None else sigma_train,
            stimulus_regime=stimulus_regime,
            angle_mode=angle_mode,
        )

    # ------------------------------------------------------- observation model
    def _objects(self, object_set: str) -> np.ndarray:
        """Latent identities for the requested task (``"A"`` or ``"B"``)."""
        if object_set == "A":
            return self.Z
        if object_set == "B":
            return self.Z_b
        raise ValueError(f"object_set must be 'A' or 'B', got {object_set!r}")

    def _project(self, Z: np.ndarray) -> np.ndarray:
        """Noise-free pre-activation observation ``tanh(W z + b)`` for ``Z``."""
        # Z: (N, d_z); W: (d_x, d_z) -> (N, d_x)
        return np.tanh(Z @ self.W.T + self.b)

    def clean_observation(self, object_set: str = "A") -> np.ndarray:
        """Noise-free observations (sigma=0); used for evaluation / extraction."""
        return self._project(self._objects(object_set))

    def noisy_observation(self, rng: np.random.Generator, object_set: str = "A") -> np.ndarray:
        """Observations with fresh additive noise ``eps ~ N(0, sigma_train^2 I)``."""
        x = self._project(self._objects(object_set))
        if self.sigma_train > 0:
            x = x + rng.standard_normal(x.shape) * self.sigma_train
        return x

    # -------------------------------------------------------------- targets
    def base_angles(self, object_set: str = "A") -> np.ndarray:
        """Per-object reference angle, derived from the input (patch Section 3).

        ``theta_i = atan2(z_{i,1}, z_{i,0})`` over the first two latent coordinates.
        For ``z ~ N(0, I)`` this is uniform on the circle, and crucially it is a
        smooth function of the (invertible) observation ``tanh(W z)`` - so an
        abstract input->angle rule exists and can transfer to novel objects.
        """
        Z = self._objects(object_set)
        return np.arctan2(Z[:, 1], Z[:, 0])

    def targets(self, similarity: float, object_set: str = "A") -> np.ndarray:
        """Target vectors for task ``T(s)``: ``(cos(theta+s), sin(theta+s))``."""
        theta = self.base_angles(object_set) + float(similarity)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # -------------------------------------------------------------- sampling
    def epoch_order(self, rng: np.random.Generator) -> np.ndarray:
        """Random permutation of the N object indices (one presentation each)."""
        return rng.permutation(CONSTANTS.n_objects)

    def min_pairwise_sep_deg(self, object_set: str = "A") -> float:
        """Minimum wrapped angular gap between distinct object base angles (degrees)."""
        theta = np.sort(self.base_angles(object_set))
        gaps = np.diff(np.concatenate([theta, theta[:1] + 2.0 * np.pi]))
        return float(np.degrees(gaps.min()))


def feature_routing_slices(d_obs: int, n_modules: int) -> list[slice]:
    """Contiguous disjoint feature slices, one per module (Section 3.3)."""
    if d_obs % n_modules != 0:
        raise ValueError(f"d_obs {d_obs} not divisible by n_modules {n_modules}")
    step = d_obs // n_modules
    return [slice(m * step, (m + 1) * step) for m in range(n_modules)]
