"""Representational analyses (Part IV).

Formulas are re-implemented locally (no cross-project imports) but numerically
matched to the corresponding ``a1b2_modular`` functions for comparability:
participation ratio, principal angles, linear CKA, hidden drift, plus PCA helpers
and representational similarity matrices.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


# ----------------------------------------------------------------- dimensionality
def participation_ratio(hids: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2) of the covariance eigenvalues.

    Mirrors ``a1b2.analysis.transfer_interference.compute_participation_ratio``.
    """
    hids = np.asarray(hids, dtype=float)
    if hids.ndim != 2 or hids.shape[0] == 0:
        return float("nan")
    x = hids - hids.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    evals = np.asarray(np.linalg.eigvalsh(cov), dtype=float)
    evals = np.clip(evals, 0.0, None)
    s1 = evals.sum()
    s2 = (evals ** 2).sum()
    if s2 <= 0:
        return float("nan")
    return float(s1 * s1 / s2)


def n_components_for_variance(hids: np.ndarray, thresholds=(0.9, 0.99)) -> dict:
    """Minimal #PCs to reach each cumulative-variance threshold."""
    hids = np.asarray(hids, dtype=float)
    out = {thr: 0 for thr in thresholds}
    if hids.ndim != 2 or hids.shape[0] < 2:
        return out
    pca = PCA().fit(hids)
    cum = np.cumsum(pca.explained_variance_ratio_)
    for thr in thresholds:
        idx = int(np.argmax(cum >= thr)) if np.any(cum >= thr) else len(cum) - 1
        out[thr] = idx + 1
    return out


# --------------------------------------------------------------- principal angles
def principal_angles(A: np.ndarray, B: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Principal angles (degrees) between the top-``k`` PCA subspaces of A and B.

    Mirrors ``a1b2.analysis.transfer_interference.compute_principal_angle``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        return np.full(n_components, np.nan)
    n_feat = A.shape[1]
    na = int(min(n_components, A.shape[0], n_feat))
    nb = int(min(n_components, B.shape[0], n_feat))
    if na < 1 or nb < 1:
        return np.full(n_components, np.nan)
    pa = PCA(n_components=na).fit(A)
    pb = PCA(n_components=nb).fit(B)
    M = pa.components_ @ pb.components_.T
    sv = np.linalg.svd(M, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(sv, -1.0, 1.0)))
    return angles


# ------------------------------------------------------------------------- CKA
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment. Mirrors ``a1b2.analysis.correlations.CKA``."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.trace(Y @ Y.T @ X @ X.T)
    den = np.sqrt(np.trace(X @ X.T @ X @ X.T) * np.trace(Y @ Y.T @ Y @ Y.T))
    if den == 0:
        return 0.0
    return float(num / den)


# ------------------------------------------------------------------------- RSA
def similarity_matrix(hids: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Object-by-object representational similarity matrix.

    ``metric`` is ``"cosine"`` or ``"correlation"``.
    """
    hids = np.asarray(hids, dtype=float)
    if metric == "correlation":
        hids = hids - hids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(hids, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = hids / norms
    return unit @ unit.T


# ------------------------------------------------------------------------ drift
def hidden_drift(pre: np.ndarray, post: np.ndarray, metric: str = "l2") -> float:
    """Mean per-object distance between two representation matrices.

    ``metric`` is ``"l2"`` (mirrors ``compute_hidden_drift``) or ``"cosine"``.
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if pre.shape != post.shape:
        raise ValueError(f"shape mismatch {pre.shape} vs {post.shape}")
    if metric == "l2":
        return float(np.mean(np.linalg.norm(post - pre, axis=1)))
    if metric == "cosine":
        pn = pre / np.clip(np.linalg.norm(pre, axis=1, keepdims=True), 1e-12, None)
        qn = post / np.clip(np.linalg.norm(post, axis=1, keepdims=True), 1e-12, None)
        return float(np.mean(1.0 - np.sum(pn * qn, axis=1)))
    raise ValueError(f"unknown metric {metric!r}")


def per_object_drift(pre: np.ndarray, post: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Per-object distance vector between two representation matrices."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if metric == "l2":
        return np.linalg.norm(post - pre, axis=1)
    if metric == "cosine":
        pn = pre / np.clip(np.linalg.norm(pre, axis=1, keepdims=True), 1e-12, None)
        qn = post / np.clip(np.linalg.norm(post, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - np.sum(pn * qn, axis=1)
    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------- angular error
def angular_error_deg(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-object absolute angular error (degrees) on the unit circle.

    Both ``preds`` and ``targets`` are ``(N, 2)`` arrays of ``(cos, sin)``-like
    vectors. This is the task-appropriate analog of "accuracy" for the continuous
    angular regression target (Part II Section 9-10): there is no classification
    accuracy in this benchmark.
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    pa = np.arctan2(preds[:, 1], preds[:, 0])
    ta = np.arctan2(targets[:, 1], targets[:, 0])
    diff = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return np.degrees(np.abs(diff))


# ---------------------------------------------------------------- rule shift
def rule_shift(preds: np.ndarray, targets: np.ndarray, similarity: float,
               tol_deg: float = 1.0) -> dict:
    """Directional interference / rule-shift metric (Holton von Mises analog).

    Quantifies how far the network's reference-task (A) predictions have been
    *pulled toward task B's rule* after the B phase. It is meant to be evaluated
    on the **A2-start** extraction record (``phase == "A2"``, ``epoch == 0``),
    where ``preds`` are the model's predictions on the clean A observations and
    ``targets`` are the A-task ring ``(cos theta_i, sin theta_i)``. Task B applies
    a rigid rotation by ``similarity`` (s), so B's target for object ``i`` sits at
    angle ``theta_i + s``.

    For each object we compute the signed angular displacement of the prediction
    away from its A target, ``delta_i = wrap(phi_i - theta_i)``, and project it
    onto the B direction ``sign(s)``. This is the per-object, directional analog
    of the von Mises pull reported by Holton et al.: a positive projection means
    that object's prediction drifted toward B's rule rather than staying on A.

    Returns a dict with:
      - ``frac_toward_B``    : fraction of objects pulled toward B by > ``tol_deg``.
      - ``mean_signed_shift_deg`` : mean signed displacement toward B (degrees);
        0 = A preserved, ~``|s|`` (deg) = fully on B.
      - ``pull_fraction``    : ``mean(delta_i) / s`` in [~0, ~1]; 0 = no pull,
        1 = predictions sit exactly on B's rule.

    The metric is **only defined for s != 0** (i.e. the Near / Far conditions);
    for the Same condition (``s == 0``) all fields are ``nan`` because there is no
    distinct B rule to be pulled toward.
    """
    s = float(similarity)
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if s == 0.0:
        return {
            "frac_toward_B": float("nan"),
            "mean_signed_shift_deg": float("nan"),
            "pull_fraction": float("nan"),
        }
    phi = np.arctan2(preds[:, 1], preds[:, 0])        # predicted angle
    theta = np.arctan2(targets[:, 1], targets[:, 0])  # A-target angle
    # Signed displacement of the prediction relative to its A target, wrapped to (-pi, pi].
    delta = np.arctan2(np.sin(phi - theta), np.cos(phi - theta))
    direction = np.sign(s)
    toward_b = delta * direction                       # > 0 means pulled toward B
    tol = np.radians(tol_deg)
    return {
        "frac_toward_B": float(np.mean(toward_b > tol)),
        "mean_signed_shift_deg": float(np.degrees(np.mean(toward_b))),
        "pull_fraction": float(np.mean(delta) / s),
    }


# ------------------------------------------------------------------------- PCA
def shared_pca_three_phase(R_a1, R_b, R_a2, n_components: int = 2):
    """Fit one PCA on stacked [A1; B; A2] reps; return (pca, proj_a1, proj_b, proj_a2)."""
    R_a1 = np.asarray(R_a1, dtype=float)
    R_b = np.asarray(R_b, dtype=float)
    R_a2 = np.asarray(R_a2, dtype=float)
    stacked = np.vstack([R_a1, R_b, R_a2])
    n = min(n_components, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n).fit(stacked)
    return pca, pca.transform(R_a1), pca.transform(R_b), pca.transform(R_a2)
