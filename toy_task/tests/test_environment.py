"""Environment determinism and task geometry (Stage-1 gate)."""

import numpy as np

from toy_task.config import (
    ANGLE_EVEN,
    ANGLE_RANDOM,
    CONSTANTS,
    SIMILARITY_GRID,
    STIMULUS_SHARED,
    STIMULUS_NOVEL,
)
from toy_task.environment import Environment, feature_routing_slices


def test_environment_deterministic_per_seed():
    e1 = Environment.from_seed(3)
    e2 = Environment.from_seed(3)
    assert np.allclose(e1.Z, e2.Z)
    assert np.allclose(e1.W, e2.W)
    assert np.allclose(e1.clean_observation(), e2.clean_observation())


def test_different_seeds_differ():
    e1 = Environment.from_seed(0)
    e2 = Environment.from_seed(1)
    assert not np.allclose(e1.Z, e2.Z)


def test_observation_shapes_and_W_variance():
    e = Environment.from_seed(0)
    assert e.Z.shape == (CONSTANTS.n_objects, CONSTANTS.d_latent)
    assert e.W.shape == (CONSTANTS.d_obs, CONSTANTS.d_latent)
    assert np.allclose(e.b, 0.0)
    x = e.clean_observation()
    assert x.shape == (CONSTANTS.n_objects, CONSTANTS.d_obs)
    assert np.all(np.abs(x) <= 1.0)  # tanh range


def test_targets_on_unit_circle_and_rigid():
    e = Environment.from_seed(0)
    base = e.targets(0.0)
    for s in SIMILARITY_GRID:
        Y = e.targets(s)
        assert np.allclose(np.linalg.norm(Y, axis=1), 1.0)
    # Rigid rotation: pairwise angles between objects preserved across s.
    def gaps(Y):
        ang = np.arctan2(Y[:, 1], Y[:, 0])
        return np.diff(np.unwrap(ang))
    assert np.allclose(gaps(base), gaps(e.targets(0.7)))


def test_clean_observation_invariant_to_similarity():
    e = Environment.from_seed(0)
    x = e.clean_observation()
    # Targets change with s, observations do not.
    assert not np.allclose(e.targets(0.0), e.targets(1.0))
    assert np.allclose(x, e.clean_observation())


def test_noise_only_in_training():
    e = Environment.from_seed(0)
    rng = np.random.default_rng(0)
    noisy = e.noisy_observation(rng)
    clean = e.clean_observation()
    assert not np.allclose(noisy, clean)  # sigma_train > 0


def test_even_angles_match_patch_ring():
    e = Environment.from_seed(0, angle_mode=ANGLE_EVEN)
    expected = 2.0 * np.pi * np.arange(CONSTANTS.n_objects) / CONSTANTS.n_objects
    assert np.allclose(np.sort(e.base_angles() % (2 * np.pi)), expected)


def test_random_angles_are_input_derived():
    e = Environment.from_seed(0, angle_mode=ANGLE_RANDOM)
    expected = np.arctan2(e.Z[:, 1], e.Z[:, 0])
    assert np.allclose(e.base_angles(), expected)
    legacy_ring = 2.0 * np.pi * np.arange(CONSTANTS.n_objects) / CONSTANTS.n_objects
    assert not np.allclose(np.sort(e.base_angles() % (2 * np.pi)), legacy_ring)


def test_default_angle_mode_is_even():
    e = Environment.from_seed(0)
    expected = 2.0 * np.pi * np.arange(CONSTANTS.n_objects) / CONSTANTS.n_objects
    assert np.allclose(np.sort(e.base_angles() % (2 * np.pi)), expected)


def test_shared_regime_reuses_task_a_objects():
    e = Environment.from_seed(0, stimulus_regime=STIMULUS_SHARED)
    assert np.allclose(e.Z_b, e.Z)
    assert np.allclose(e.clean_observation("B"), e.clean_observation("A"))
    assert np.allclose(e.targets(0.3, "B"), e.targets(0.3, "A"))


def test_novel_regime_uses_fresh_objects_but_shares_task_a():
    shared = Environment.from_seed(7, stimulus_regime=STIMULUS_SHARED)
    novel = Environment.from_seed(7, stimulus_regime=STIMULUS_NOVEL)
    # Task A (and W) are bit-identical across regimes for the same seed.
    assert np.allclose(shared.Z, novel.Z)
    assert np.allclose(shared.W, novel.W)
    # Task B objects are a fresh, distinct draw under the novel regime.
    assert not np.allclose(novel.Z_b, novel.Z)
    assert not np.allclose(novel.clean_observation("B"), novel.clean_observation("A"))


def test_novel_regime_is_deterministic():
    e1 = Environment.from_seed(5, stimulus_regime=STIMULUS_NOVEL)
    e2 = Environment.from_seed(5, stimulus_regime=STIMULUS_NOVEL)
    assert np.allclose(e1.Z_b, e2.Z_b)


def test_invalid_regime_rejected():
    import pytest

    with pytest.raises(ValueError):
        Environment.from_seed(0, stimulus_regime="bogus")


def test_feature_routing_slices():
    sl = feature_routing_slices(8, 2)
    assert sl == [slice(0, 4), slice(4, 8)]
