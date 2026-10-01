"""Neighbor-restricted denoising score entropy for a tiny count grid.

The network outputs a positive score for each coordinate. The training target
is the conditional ratio specified for this kernel:

    s(xt, j, +1) = q(xt with coordinate j decreased by 1 | x0) / q(xt | x0)

when xt_j - 1 is on the grid. Coordinates whose +1 neighbor would leave the
grid are not trained. The loss is the Bregman score entropy

    L(s, t) = s - t log s - t

whose gap L(s, t) - L(t, t) is nonnegative and zero at s = t.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from disco.scores import GRID, chain_log_joint, sample_joint

SIGMAS = (0.1, 0.3)


def _expand_sigma(sigma, like):
    """Broadcast a scalar or per-example sigma across coordinates."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    like = jnp.asarray(like)
    if sigma.ndim == 0 or sigma.shape == like.shape:
        return sigma
    if sigma.shape == (like.shape[0],) and like.ndim == 2:
        return sigma[:, None]
    return sigma


def coord_transition_prob(y, x0, sigma, grid):
    """Coordinate-wise noise probability q(y | x0), after boundary renormalization.

    Unnormalized mass is 1 - sigma on a stay, sigma/2 on a legal +1 step, and
    sigma/2 on a legal -1 step. Mass that would leave the grid is dropped and
    the remaining mass is renormalized.
    """
    y = jnp.asarray(y)
    x0 = jnp.asarray(x0)
    sigma = _expand_sigma(sigma, x0)
    stay = 1.0 - sigma
    down = jnp.where(x0 > 0, sigma / 2.0, 0.0)
    up = jnp.where(x0 + 1 < grid, sigma / 2.0, 0.0)
    normalizer = stay + down + up
    on_grid = (y >= 0) & (y < grid)
    prob = jnp.where(
        y == x0,
        stay / normalizer,
        jnp.where(y == x0 - 1, down / normalizer, jnp.where(y == x0 + 1, up / normalizer, 0.0)),
    )
    return jnp.where(on_grid, prob, 0.0)


def corrupt(key, x0, sigma, grid):
    """One independent noise step per coordinate. xt stays on the grid."""
    x0 = jnp.asarray(x0)
    sigma = _expand_sigma(sigma, x0)
    stay = 1.0 - sigma
    down = jnp.where(x0 > 0, sigma / 2.0, 0.0)
    up = jnp.where(x0 + 1 < grid, sigma / 2.0, 0.0)
    normalizer = stay + down + up
    p_down = down / normalizer
    p_stay = stay / normalizer
    draw = jax.random.uniform(key, x0.shape)
    step = jnp.where(draw < p_down, -1, jnp.where(draw < p_down + p_stay, 0, 1))
    return x0 + step


def conditional_target(xt, x0, sigma, grid):
    """Conditional concrete-score target and its defined mask.

    target = q(xt_j - 1 | x0_j) / q(xt_j | x0_j)
    defined when xt_j - 1 is on the grid and a +1 step from xt_j stays on the
    grid, so an off-grid +1 score is not trained.
    """
    q_current = coord_transition_prob(xt, x0, sigma, grid)
    q_back = coord_transition_prob(xt - 1, x0, sigma, grid)
    ratio = jnp.where(q_current > 0, q_back / jnp.maximum(q_current, 1e-30), 0.0)
    defined = (xt >= 1) & (xt + 1 < grid)
    return ratio, defined


def score_entropy(s, target):
    """L(s, t) = s - t log s - t. Minimum over s > 0 is at s = t."""
    s = jnp.asarray(s)
    target = jnp.asarray(target)
    return s - target * jnp.log(s) - target


def bregman_gap(s, target):
    """L(s, t) - L(t, t). Nonnegative, and zero when s = t.

    The t = 0 case uses the continuous extension L(s, 0) - 0 = s.
    """
    s = jnp.asarray(s)
    target = jnp.asarray(target)
    positive = target > 0
    safe = jnp.maximum(target, 1e-30)
    gap_positive = s - target - target * jnp.log(s / safe)
    gap_zero = s
    return jnp.where(positive, gap_positive, gap_zero)


def _init_mlp(key, in_dim, width, out_dim):
    k1, k2 = jax.random.split(key)
    params = {
        "w1": jax.random.normal(k1, (in_dim, width)) * jnp.sqrt(2.0 / in_dim),
        "b1": jnp.zeros((width,)),
        "w2": jax.random.normal(k2, (width, out_dim)) * jnp.sqrt(1.0 / width),
        # A zero output bias sits near the mean target, so the gap has little
        # room to fall. A small negative bias starts the score too low.
        "b2": jnp.full((out_dim,), -1.0),
    }
    return params


def _featurize(xt, sigma, grid):
    one_hot = jax.nn.one_hot(xt, grid, dtype=jnp.float64).reshape(xt.shape[0], -1)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    if sigma.ndim == 0:
        sigma_col = jnp.full((xt.shape[0], 1), sigma)
    else:
        sigma_col = sigma.reshape(xt.shape[0], 1)
    return jnp.concatenate([one_hot, sigma_col], axis=1)


def _score(params, xt, sigma, grid):
    hidden = jax.nn.silu(_featurize(xt, sigma, grid) @ params["w1"] + params["b1"])
    raw = hidden @ params["w2"] + params["b2"]
    return jax.nn.softplus(raw) + 1e-6


def _masked_mean(values, defined):
    defined = defined.astype(values.dtype)
    return jnp.sum(values * defined) / jnp.maximum(jnp.sum(defined), 1.0)


def batch_loss(params, xt, x0, sigma, grid):
    """Mean Bregman gap on coordinates where the target is defined.

    The gap has the same parameter gradient as L, and it is nonnegative, so a
    drop is a real reduction rather than a shift of a negative constant.
    """
    s = _score(params, xt, sigma, grid)
    target, defined = conditional_target(xt, x0, sigma, grid)
    return _masked_mean(bregman_gap(s, target), defined)


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_apply(params, grads, state, lr):
    t = state["t"] + 1
    m = jax.tree.map(lambda mom, g: 0.9 * mom + 0.1 * g, state["m"], grads)
    v = jax.tree.map(lambda vel, g: 0.999 * vel + 0.001 * (g * g), state["v"], grads)
    m_hat = jax.tree.map(lambda mom: mom / (1.0 - jnp.power(0.9, t)), m)
    v_hat = jax.tree.map(lambda vel: vel / (1.0 - jnp.power(0.999, t)), v)
    updated = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8),
        params,
        m_hat,
        v_hat,
    )
    return updated, {"m": m, "v": v, "t": t}


def _sample_sigma(key, batch):
    choice = jax.random.randint(key, (batch,), 0, 2)
    return jnp.where(choice == 0, SIGMAS[0], SIGMAS[1]).astype(jnp.float64)


def _make_eval_batch(key, data, grid, size):
    k_idx, k_sigma, k_noise = jax.random.split(key, 3)
    index = jax.random.randint(k_idx, (size,), 0, data.shape[0])
    x0 = data[index]
    sigma = _sample_sigma(k_sigma, size)
    xt = corrupt(k_noise, x0, sigma, grid)
    return x0, xt, sigma


def train_score(steps=300, seed=0, batch_size=128, width=64, grid=GRID, n_data=4096, eval_size=1024, lr=1e-2):
    """Train the neighbor-restricted score network. Returns start and end loss.

    Loss is the mean Bregman gap of L on a fixed evaluation batch, measured
    before the first update and after `steps` Adam updates. Seed 0 fixes the
    data, the initialization, the noise, and the evaluation batch.
    """
    key = jax.random.PRNGKey(seed)
    key_data, key_init, key_eval, key_steps = jax.random.split(key, 4)
    log_joint = chain_log_joint(grid=grid)
    data = jnp.asarray(sample_joint(key_data, log_joint, n_data))
    d = data.shape[1]
    params = _init_mlp(key_init, in_dim=d * grid + 1, width=width, out_dim=d)
    opt_state = _adam_init(params)
    eval_x0, eval_xt, eval_sigma = _make_eval_batch(key_eval, data, grid, eval_size)

    def loss_on_eval(p):
        return batch_loss(p, eval_xt, eval_x0, eval_sigma, grid)

    loss_start = float(loss_on_eval(params))

    @jax.jit
    def step(p, state, x0, sigma, noise_key):
        xt = corrupt(noise_key, x0, sigma, grid)

        def loss_fn(pp):
            return batch_loss(pp, xt, x0, sigma, grid)

        loss, grads = jax.value_and_grad(loss_fn)(p)
        p, state = _adam_apply(p, grads, state, lr)
        return p, state, loss

    for t in range(steps):
        key_steps, k_idx, k_sigma, k_noise = jax.random.split(key_steps, 4)
        index = jax.random.randint(k_idx, (batch_size,), 0, data.shape[0])
        x0 = data[index]
        sigma = _sample_sigma(k_sigma, batch_size)
        params, opt_state, _train_loss = step(params, opt_state, x0, sigma, k_noise)

    loss_end = float(loss_on_eval(params))
    return {
        "loss_start": loss_start,
        "loss_end": loss_end,
        "steps": steps,
        "params": params,
    }
