"""Analysis formula sanity checks on tiny synthetic inputs."""

import numpy as np

from toy_task.analysis import (
    participation_ratio,
    principal_angles,
    linear_cka,
    similarity_matrix,
    hidden_drift,
    per_object_drift,
    rule_shift,
)


def _ring(angles: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def test_participation_ratio_rank_one():
    # Data varying along a single direction -> PR ~ 1.
    base = np.random.default_rng(0).standard_normal((20, 1))
    direction = np.array([[1.0, 2.0, -1.0]])
    X = base @ direction
    assert abs(participation_ratio(X) - 1.0) < 0.1


def test_participation_ratio_isotropic():
    X = np.random.default_rng(1).standard_normal((500, 5))
    pr = participation_ratio(X)
    assert 3.5 < pr <= 5.0  # close to full dimensionality


def test_principal_angles_identical_subspaces():
    X = np.random.default_rng(2).standard_normal((10, 6))
    ang = principal_angles(X, X, n_components=2)
    assert np.all(ang < 1e-3)  # identical subspaces -> ~0 degrees


def test_cka_self_is_one():
    X = np.random.default_rng(3).standard_normal((8, 12))
    assert abs(linear_cka(X, X) - 1.0) < 1e-8


def test_similarity_matrix_diagonal_ones():
    X = np.random.default_rng(4).standard_normal((8, 10))
    S = similarity_matrix(X, metric="cosine")
    assert np.allclose(np.diag(S), 1.0)
    assert S.shape == (8, 8)


def test_drift_zero_when_identical():
    X = np.random.default_rng(5).standard_normal((8, 10))
    assert hidden_drift(X, X) == 0.0
    assert np.allclose(per_object_drift(X, X), 0.0)


# ---------------------------------------------------------------- rule shift
def test_rule_shift_same_condition_is_nan():
    theta = 2.0 * np.pi * np.arange(8) / 8
    targ = _ring(theta)
    out = rule_shift(targ, targ, similarity=0.0)
    assert all(np.isnan(v) for v in out.values())


def test_rule_shift_no_pull_when_predictions_on_A():
    # Predictions sit exactly on the A target -> no pull toward B.
    theta = 2.0 * np.pi * np.arange(8) / 8
    targ = _ring(theta)
    out = rule_shift(targ, targ, similarity=0.5)
    assert out["frac_toward_B"] == 0.0
    assert abs(out["mean_signed_shift_deg"]) < 1e-6
    assert abs(out["pull_fraction"]) < 1e-9


def test_rule_shift_full_pull_when_predictions_on_B():
    # Predictions sit exactly on B's rule (theta + s) -> fully pulled.
    s = 0.5
    theta = 2.0 * np.pi * np.arange(8) / 8
    targ = _ring(theta)
    preds = _ring(theta + s)
    out = rule_shift(preds, targ, similarity=s)
    assert out["frac_toward_B"] == 1.0
    assert abs(out["pull_fraction"] - 1.0) < 1e-9
    assert abs(out["mean_signed_shift_deg"] - np.degrees(s)) < 1e-6


def test_rule_shift_direction_sign_independent():
    # A negative-s rotation pulled halfway should also register as toward B.
    s = -0.4
    theta = 2.0 * np.pi * np.arange(8) / 8
    targ = _ring(theta)
    preds = _ring(theta + s / 2)
    out = rule_shift(preds, targ, similarity=s)
    assert out["frac_toward_B"] == 1.0
    assert abs(out["pull_fraction"] - 0.5) < 1e-9
