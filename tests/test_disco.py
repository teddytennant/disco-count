"""Oracle CCS/OCS recovery and the score-entropy trainer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from disco.estimate import (
    bregman_gap,
    conditional_target,
    coord_transition_prob,
    corrupt,
    score_entropy,
    train_score,
)
from disco.scores import (
    PARENT_TAU,
    bivariate_log_joint,
    ccs_on_subset,
    chain_log_joint,
    curvature,
    disco_order,
    log_concrete_plus,
    ocs_given_order,
    rank_transform,
    sample_joint,
    select_parents,
    truncated_poisson_log,
)


def test_rank_transform_ties_and_log1p_invariance():
    column = np.array([1.0, 2.0, 2.0, 4.0])
    other = np.array([0.0, 3.0, 3.0, 1.0])
    raw = np.stack([column, other], axis=1)
    ranks = rank_transform(raw)
    expected = np.array([1.0, 2.5, 2.5, 4.0]) / 5.0
    assert np.allclose(ranks[:, 0], expected)
    assert np.all((ranks > 0.0) & (ranks < 1.0))
    transformed = raw.copy()
    transformed[:, 0] = np.log1p(transformed[:, 0])
    ranks_transformed = rank_transform(transformed)
    assert np.allclose(ranks[:, 0], ranks_transformed[:, 0])
    assert np.allclose(ranks[:, 1], ranks_transformed[:, 1])


def test_concrete_score_matches_finite_difference_ell():
    log_joint = chain_log_joint()
    cells = [(0, 1, 2), (1, 0, 3), (2, 2, 1), (3, 1, 0), (0, 4, 2)]
    for state in cells:
        for j in range(3):
            if state[j] + 1 >= 5:
                assert np.isnan(log_concrete_plus(log_joint, state, j))
                continue
            shifted = list(state)
            shifted[j] += 1
            concrete = np.exp(log_joint[tuple(shifted)] - log_joint[tuple(state)])
            ell = log_concrete_plus(log_joint, state, j)
            assert np.allclose(ell, np.log(concrete))
            for i in range(3):
                h = curvature(log_joint, state, i, j)
                neighbor = list(state)
                if neighbor[i] + 1 >= 5:
                    assert np.isnan(h)
                    continue
                neighbor[i] += 1
                ell_neighbor = log_concrete_plus(log_joint, neighbor, j)
                if not np.isfinite(ell_neighbor):
                    assert np.isnan(h)
                else:
                    assert np.allclose(h, ell_neighbor - ell)


def test_chain_order_and_parents():
    log_joint = chain_log_joint()
    source = np.exp(truncated_poisson_log())
    assert np.all(source > 0.0)
    assert not np.allclose(source, np.full(5, 0.2))
    samples = sample_joint(jax.random.PRNGKey(0), log_joint, 4000)

    full = ccs_on_subset(samples, log_joint, [0, 1, 2])
    assert full[2] < full[0]
    assert full[2] < full[1]
    assert full[2] < 1e-8
    assert full[0] > 1e-3
    assert full[1] > 1e-3

    remaining = ccs_on_subset(samples, log_joint, [0, 1])
    assert remaining[1] < remaining[0]
    assert remaining[1] < 1e-8
    assert remaining[0] > 1e-3

    order = disco_order(samples, log_joint)
    assert order == [0, 1, 2]

    ocs = ocs_given_order(samples, log_joint, order)
    assert ocs[(0, 1)] > ocs[(0, 2)]
    assert ocs[(1, 2)] > ocs[(0, 2)]
    assert ocs[(0, 2)] < 1e-8
    assert ocs[(0, 1)] > PARENT_TAU
    assert ocs[(1, 2)] > PARENT_TAU
    assert select_parents(order, ocs, tau=PARENT_TAU) == [(0, 1), (1, 2)]


def test_bivariate_nonlinear_eta_child_zero_source_positive():
    """Nonlinear eta: the child has zero CCS and the source has positive CCS.

    That is the bivariate case of the paper (Theorem 6): for X -> Y,
    S_Y = 0 < S_X exactly when eta is nonlinear. Ordering then puts the
    source first.
    """
    log_joint = bivariate_log_joint(source=0)
    samples = sample_joint(jax.random.PRNGKey(0), log_joint, 4000)
    scores = ccs_on_subset(samples, log_joint, [0, 1])
    assert scores[1] < 1e-8
    assert scores[0] > 1e-3
    assert disco_order(samples, log_joint) == [0, 1]

    reversed_joint = bivariate_log_joint(source=1)
    reversed_samples = sample_joint(jax.random.PRNGKey(1), reversed_joint, 4000)
    reversed_scores = ccs_on_subset(reversed_samples, reversed_joint, [0, 1])
    assert reversed_scores[0] < 1e-8
    assert reversed_scores[1] > 1e-3
    assert disco_order(reversed_samples, reversed_joint) == [1, 0]


def test_score_entropy_gap_nonnegative_and_zero_when_matched():
    key = jax.random.PRNGKey(0)
    s = jax.random.uniform(key, (24,), minval=0.05, maxval=6.0)
    t = jax.random.uniform(jax.random.fold_in(key, 1), (24,), minval=0.05, maxval=6.0)
    gap = score_entropy(s, t) - score_entropy(t, t)
    assert np.all(np.asarray(gap) >= -1e-8)
    assert np.allclose(score_entropy(t, t) - score_entropy(t, t), 0.0)
    assert np.allclose(bregman_gap(t, t), 0.0)
    assert np.allclose(bregman_gap(s, t), gap)
    assert float(score_entropy(1.0, 1.0)) == pytest.approx(0.0)


def test_target_is_kernel_ratio_not_a_network():
    sigma = 0.1
    grid = 5
    xt = np.array([[2, 0, 4]], dtype=np.int32)
    x0 = np.array([[2, 0, 1]], dtype=np.int32)
    target, defined = conditional_target(xt, x0, sigma, grid)
    # Interior stay: q(xt-1 | x0) / q(xt | x0) = (sigma/2) / (1-sigma).
    assert bool(defined[0, 0])
    assert float(target[0, 0]) == pytest.approx((sigma / 2.0) / (1.0 - sigma))
    # xt_j = 0: shifted-back state is off the grid, so the target is undefined.
    assert not bool(defined[0, 1])
    # xt_j = M-1: a +1 step leaves the grid, so that coordinate is not trained.
    assert not bool(defined[0, 2])

    # Boundary renormalization: x0 = 0, xt = 1, ratio of stay to up is 0.9 / 0.05.
    xt_b = np.array([[1]], dtype=np.int32)
    x0_b = np.array([[0]], dtype=np.int32)
    target_b, defined_b = conditional_target(xt_b, x0_b, sigma, grid)
    assert bool(defined_b[0, 0])
    assert float(target_b[0, 0]) == pytest.approx(0.9 / 0.05)
    q_stay = float(coord_transition_prob(0, 0, sigma, grid))
    q_up = float(coord_transition_prob(1, 0, sigma, grid))
    assert q_stay == pytest.approx(0.9 / 0.95)
    assert q_up == pytest.approx(0.05 / 0.95)


def test_corrupt_respects_grid_and_boundary():
    x0 = jnp.zeros((80, 3), dtype=jnp.int32)
    xt = np.asarray(corrupt(jax.random.PRNGKey(0), x0, 0.3, 5))
    assert np.all(xt >= 0)
    assert np.all(xt <= 1)


def test_train_loss_drops_in_40_steps():
    result = train_score(steps=40, seed=0)
    assert result["loss_end"] < result["loss_start"]
    assert np.isfinite(result["loss_start"])
    assert np.isfinite(result["loss_end"])
