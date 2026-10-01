"""Population CCS, OCS, ordering, parent selection, and the rank transform.

Curvature follows paper equation (4): H_ij(x) = ell_j(x + e_i) - ell_j(x),
where ell_j is the log concrete score of a +1 shift. CCS is the grouped
conditional variance in equation (5). OCS is the mean absolute off-diagonal
curvature in Definition 2, scored only on predecessors and on the predecessor
marginal.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Separates true parents from the non-edge on the M=5 chain in the tests.
# Parent OCS values sit above 0.1 and the missing edge is numerically zero.
PARENT_TAU = 0.05

GRID = 5
POISSON_LAM = 1.5


def _logsumexp(values, axis=None):
    values = np.asarray(values, dtype=np.float64)
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value)
    total = np.log(np.sum(stable, axis=axis, keepdims=True)) + max_value
    if axis is None:
        return float(np.reshape(total, ()))
    return np.squeeze(total, axis=axis)


def _log_factorial(grid):
    log_fact = np.zeros(grid, dtype=np.float64)
    for k in range(1, grid):
        log_fact[k] = log_fact[k - 1] + np.log(k)
    return log_fact


def truncated_poisson_log(grid=GRID, lam=POISSON_LAM):
    """Log pmf of a Poisson(lam) truncated to {0, ..., grid-1} and renormalized."""
    counts = np.arange(grid, dtype=np.float64)
    log_p = -lam + counts * np.log(lam) - _log_factorial(grid)
    return log_p - _logsumexp(log_p)


def _glm_conditional_log(eta, grid, log_base):
    counts = np.arange(grid, dtype=np.float64)
    log_unnorm = log_base + counts * float(eta)
    return log_unnorm - _logsumexp(log_unnorm)


def chain_log_joint(grid=GRID, lam=POISSON_LAM, use_factorial_base=False):
    """Log joint of the identifiable 3-node chain X0 -> X1 -> X2.

    Source X0 is a truncated Poisson. Non-sources are semiparametric GLM
    conditionals (paper equation 3) with nonlinear canonical parameters
    eta_1(x0) = 0.15 x0 + 0.35 x0^2 and eta_2(x1) = 0.2 x1 + 0.4 sin(x1).
    There is no direct edge from X0 to X2. Base b is 1, or 1/x! if requested.
    """
    log_source = truncated_poisson_log(grid, lam)
    log_base = -_log_factorial(grid) if use_factorial_base else np.zeros(grid)
    log_joint = np.zeros((grid, grid, grid), dtype=np.float64)
    for x0 in range(grid):
        eta1 = 0.15 * x0 + 0.35 * (x0 ** 2)
        log_p1 = _glm_conditional_log(eta1, grid, log_base)
        for x1 in range(grid):
            eta2 = 0.2 * x1 + 0.4 * np.sin(x1)
            log_p2 = _glm_conditional_log(eta2, grid, log_base)
            log_joint[x0, x1, :] = log_source[x0] + log_p1[x1] + log_p2
    return log_joint - _logsumexp(log_joint)


def bivariate_log_joint(source=0, grid=GRID, lam=POISSON_LAM, eta_fn=None):
    """Log joint of a 2-node semiparametric GLM DAG.

    `source` is 0 or 1. The other node is the child. Default eta is the
    nonlinear map used for the first edge of the chain.
    """
    if source not in (0, 1):
        raise ValueError("source must be 0 or 1")
    if eta_fn is None:
        eta_fn = lambda x: 0.15 * x + 0.35 * (x ** 2)
    child = 1 - source
    log_source = truncated_poisson_log(grid, lam)
    log_base = np.zeros(grid, dtype=np.float64)
    log_joint = np.zeros((grid, grid), dtype=np.float64)
    counts = np.arange(grid, dtype=np.float64)
    for parent_value in range(grid):
        log_cond = _glm_conditional_log(eta_fn(parent_value), grid, log_base)
        for child_value in range(grid):
            state = [0, 0]
            state[source] = parent_value
            state[child] = child_value
            log_joint[state[0], state[1]] = log_source[parent_value] + log_cond[child_value]
    return log_joint - _logsumexp(log_joint)


def sample_joint(key, log_joint, n):
    """Inverse-CDF sample from a log joint. Returns integer states of shape (n, d)."""
    probs = np.exp(np.asarray(log_joint, dtype=np.float64))
    probs = probs / probs.sum()
    flat = probs.ravel()
    cdf = np.cumsum(flat)
    uniforms = np.asarray(jax.random.uniform(key, (n,), dtype=jnp.float64))
    index = np.searchsorted(cdf, uniforms, side="left")
    index = np.clip(index, 0, flat.size - 1)
    coords = np.stack(np.unravel_index(index, probs.shape), axis=1)
    return np.asarray(coords, dtype=np.int32)


def rank_transform(x):
    """Average rank of each column, divided by (n + 1), so values lie in (0, 1).

    Ties receive the average of the ranks they occupy. A strictly increasing
    map of a column does not change that column's ranks.
    """
    values = np.asarray(x, dtype=np.float64)
    squeezed = False
    if values.ndim == 1:
        values = values[:, None]
        squeezed = True
    if values.ndim != 2:
        raise ValueError("rank_transform expects a vector or a 2d array")
    n, d = values.shape
    ranks = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        column = values[:, j]
        less = np.sum(column[None, :] < column[:, None], axis=1)
        equal = np.sum(column[None, :] == column[:, None], axis=1)
        average_rank = less + (equal + 1.0) / 2.0
        ranks[:, j] = average_rank / (n + 1.0)
    if squeezed:
        return ranks[:, 0]
    return ranks


def _gather(log_p, states):
    log_p = np.asarray(log_p, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    flat_index = np.ravel_multi_index(states.T, log_p.shape)
    return log_p.ravel()[flat_index]


def log_concrete_plus(log_p, states, j):
    """ell_j(x) = log s_{j, x_j+1}(x) = log p(x + e_j) - log p(x).

    NaN where x_j + 1 leaves the grid. `states` is (n, d) or a single state.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    single = states.ndim == 1
    if single:
        states = states[None, :]
    grid_j = log_p.shape[j]
    xj = states[:, j]
    valid = xj + 1 < grid_j
    shifted = states.copy()
    shifted[:, j] = np.clip(xj + 1, 0, grid_j - 1)
    ell = _gather(log_p, shifted) - _gather(log_p, states)
    ell = np.where(valid, ell, np.nan)
    if single:
        return float(ell[0])
    return ell


def curvature(log_p, states, i, j):
    """H_ij(x) = ell_j(x + e_i) - ell_j(x), NaN if either ell is undefined."""
    log_p = np.asarray(log_p, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    single = states.ndim == 1
    if single:
        states = states[None, :]
    grid_i = log_p.shape[i]
    xi = states[:, i]
    can_shift = xi + 1 < grid_i
    shifted = states.copy()
    shifted[:, i] = np.clip(xi + 1, 0, grid_i - 1)
    ell = log_concrete_plus(log_p, states, j)
    ell_shifted = log_concrete_plus(log_p, shifted, j)
    valid = can_shift & np.isfinite(ell) & np.isfinite(ell_shifted)
    h = np.where(valid, ell_shifted - ell, np.nan)
    if single:
        return float(h[0])
    return h


def conditional_curvature_score(x_j, h_jj):
    """Sample-size-weighted average of within-group sample variances of H_jj.

    Groups are the values of X_j. Groups with fewer than 2 defined samples
    contribute nothing. Returns +inf when no group is usable, so that node
    is not treated as a sink.
    """
    x_j = np.asarray(x_j)
    h_jj = np.asarray(h_jj, dtype=np.float64)
    defined = np.isfinite(h_jj)
    x_j = x_j[defined]
    h_jj = h_jj[defined]
    if x_j.size == 0:
        return np.inf
    numerator = 0.0
    denominator = 0.0
    for level in np.unique(x_j):
        group = h_jj[x_j == level]
        count = int(group.size)
        if count < 2:
            continue
        variance = float(np.var(group, ddof=1))
        numerator += count * variance
        denominator += count
    if denominator == 0:
        return np.inf
    return numerator / denominator


def mean_abs_defined(h):
    """Mean of |H| over entries where the curvature is defined."""
    h = np.asarray(h, dtype=np.float64)
    defined = np.isfinite(h)
    if not np.any(defined):
        return np.nan
    return float(np.mean(np.abs(h[defined])))


def marginal_log_prob(log_joint, keep):
    """Log marginal on `keep` (original axes). Result axes follow sorted(keep)."""
    log_joint = np.asarray(log_joint, dtype=np.float64)
    keep = [int(node) for node in keep]
    drop = tuple(axis for axis in range(log_joint.ndim) if axis not in set(keep))
    if drop:
        log_marginal = _logsumexp(log_joint, axis=drop)
    else:
        log_marginal = log_joint.copy()
    return log_marginal - _logsumexp(log_marginal)


def ccs_on_subset(samples, log_joint, nodes):
    """CCS of each node in `nodes` under the exact marginal of those nodes."""
    nodes = [int(node) for node in nodes]
    order = sorted(nodes)
    log_marginal = marginal_log_prob(log_joint, order)
    projected = np.asarray(samples)[:, order]
    scores = {}
    for local, node in enumerate(order):
        h_jj = curvature(log_marginal, projected, local, local)
        scores[node] = conditional_curvature_score(projected[:, local], h_jj)
    return scores


def disco_order(samples, log_joint):
    """Source-to-sink order by repeated removal of the smallest-CCS node.

    After each removal the CCS is recomputed on the exact marginal of the
    remaining nodes. Ties break toward the smaller node index. The expected
    order is not special-cased.
    """
    samples = np.asarray(samples)
    d = samples.shape[1]
    remaining = list(range(d))
    sink_first = []
    while len(remaining) > 1:
        scores = ccs_on_subset(samples, log_joint, remaining)
        chosen = min(remaining, key=lambda node: (scores[node], node))
        sink_first.append(chosen)
        remaining.remove(chosen)
    sink_first.append(remaining[0])
    return list(reversed(sink_first))


def ocs_given_order(samples, log_joint, order):
    """OCS_{j -> i} for every predecessor j of every node i.

    Each score uses the exact marginal on {i} union the predecessors of i,
    and only samples where H_ij is defined.
    """
    samples = np.asarray(samples)
    scores = {}
    for position, child in enumerate(order):
        predecessors = list(order[:position])
        if not predecessors:
            continue
        keep = sorted(predecessors + [child])
        log_marginal = marginal_log_prob(log_joint, keep)
        projected = samples[:, keep]
        local = {node: axis for axis, node in enumerate(keep)}
        for parent in predecessors:
            h_ij = curvature(log_marginal, projected, local[child], local[parent])
            scores[(parent, child)] = mean_abs_defined(h_ij)
    return scores


def select_parents(order, ocs, tau=PARENT_TAU):
    """Keep predecessor j of i when OCS_{j -> i} > tau. Does not hardcode edges."""
    edges = []
    for position, child in enumerate(order):
        for parent in order[:position]:
            if ocs[(parent, child)] > tau:
                edges.append((parent, child))
    return edges
