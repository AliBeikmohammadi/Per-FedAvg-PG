"""Shared-structure task families and evaluation helpers.

------------------------------------------------------------
The personalized-federated-MAML value proposition is few-shot adaptation to a
new agent drawn from a task distribution with shared structure.

`make_arc_family` places all goals on a short ARC of a circle: the shared skill
is "navigate toward that arc", and one adaptation step selects the exact goal.
Held-out goals are the *midpoints* between training goals -- in-distribution but
never seen -- which is the most demanding in-distribution placement and is fully
reproducible (no random held-out selection).

Trajectory accounting (the honest x-axis)
-----------------------------------------
`per_step_cost(variant, m_in, m_h, m_out)` returns the number of environment
trajectories one local meta-gradient step consumes:
    exact  : m_in + m_h + m_out   (inner PG, curvature batch, outer PG)
    fo     : m_in + m_out         (no curvature batch)
    fedavg : m_in + m_h + m_out   (a single PG estimate with the same budget)
so figures can be plotted against trajectories actually consumed.
"""

import numpy as np
from . import neural as N


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

def make_arc_family(radius=0.8, n_train=6, n_held=3, span=(-0.5, 0.5)):
    """Goals on an arc of angular extent `span` at the given radius.

    Returns (train_goals, held_goals, info). Training goals are evenly spaced on
    the arc; held-out goals are midpoints between consecutive training goals
    (in-distribution, unseen, deterministic).
    """
    ang_train = np.linspace(span[0], span[1], n_train)
    mids = (ang_train[:-1] + ang_train[1:]) / 2.0          # n_train-1 midpoints
    if n_held > len(mids):
        raise ValueError(f"n_held={n_held} exceeds available midpoints={len(mids)}")
    # pick held-out midpoints spread across the arc (evenly indexed)
    idx = np.linspace(0, len(mids) - 1, n_held).round().astype(int)
    ang_held = mids[idx]
    g = lambda a: radius * np.array([np.cos(a), np.sin(a)])
    train_goals = [g(a) for a in ang_train]
    held_goals = [g(a) for a in ang_held]
    info = dict(radius=radius, ang_train=ang_train, ang_held=ang_held, span=span)
    return train_goals, held_goals, info


# ---------------------------------------------------------------------------
# Trajectory accounting
# ---------------------------------------------------------------------------

def per_step_cost(variant, m_in, m_h, m_out):
    if variant == 'fo':
        return m_in + m_out
    return m_in + m_h + m_out          # exact, fedavg


# ---------------------------------------------------------------------------
# Monte-Carlo evaluation and few-shot adaptation (neural)
# ---------------------------------------------------------------------------

def value_mc(goal, theta, m_eval=512, seed=999):
    """MC estimate of the (undiscounted-return-to-go from start) value of a
    policy `theta` on a single-goal task, with a fixed evaluation seed."""
    _, _, R = N.rollout(goal, theta, m_eval, np.random.default_rng(seed))
    return float(R[:, 0].mean())


def adapt_from_init(goal, theta_init, alpha, m_adapt, steps, rng):
    """Return the parameters after `steps` policy-gradient adaptation steps of
    size `alpha` from `theta_init`, using fresh batches of size `m_adapt`."""
    th = theta_init.copy()
    for _ in range(steps):
        th = th + alpha * N.pg_estimate(goal, th, m_adapt, rng)
    return th


def few_shot_from_init(goals, theta_init, alpha, m_adapt, steps,
                       m_eval=512, seed=7):
    """Average post-adaptation value over `goals`, adapting each from a common
    initialization `theta_init` with `steps` steps of batch `m_adapt`."""
    rng = np.random.default_rng(seed)
    vals = [value_mc(g, adapt_from_init(g, theta_init, alpha, m_adapt, steps, rng),
                     m_eval, seed=10_000 + i)
            for i, g in enumerate(goals)]
    return float(np.mean(vals))


def from_scratch(goals, G, m_step, beta, m_eval=512, seed=7, init_seed=123):
    """Average value after training each goal's policy from a random init with
    `G` policy-gradient steps of batch `m_step` (the independent baseline)."""
    rng = np.random.default_rng(seed)
    vals = []
    for i, g in enumerate(goals):
        th = N.init_theta(seed=init_seed)
        for _ in range(G):
            th = th + beta * N.pg_estimate(g, th, m_step, rng)
        vals.append(value_mc(g, th, m_eval, seed=20_000 + i))
    return float(np.mean(vals))


def eval_policy(goal, theta, m_eval=512, seed=999):
    """Return (expected_return, success_rate, mean_final_distance) for a policy."""
    ret = value_mc(goal, theta, m_eval, seed)
    succ, fdist = N.rollout_metrics(goal, theta, m_eval, np.random.default_rng(seed + 1))
    return ret, succ, fdist


def few_shot_metrics_from_init(goals, theta_init, alpha, m_adapt, steps,
                               m_eval=512, seed=7):
    """Average (return, success, final_dist) over goals, adapting each from a
    common init with `steps` steps of batch `m_adapt`."""
    rng = np.random.default_rng(seed)
    R, S, F = [], [], []
    for i, g in enumerate(goals):
        th = adapt_from_init(g, theta_init, alpha, m_adapt, steps, rng)
        r, s, f = eval_policy(g, th, m_eval, seed=10_000 + i)
        R.append(r); S.append(s); F.append(f)
    return float(np.mean(R)), float(np.mean(S)), float(np.mean(F))


def from_scratch_metrics(goals, G, m_step, beta, m_eval=512, seed=7, init_seed=123):
    """Average (return, success, final_dist) after training each goal from a
    random init with G policy-gradient steps of batch `m_step`."""
    rng = np.random.default_rng(seed)
    R, S, F = [], [], []
    for i, g in enumerate(goals):
        th = N.init_theta(seed=init_seed)
        for _ in range(G):
            th = th + beta * N.pg_estimate(g, th, m_step, rng)
        r, s, f = eval_policy(g, th, m_eval, seed=20_000 + i)
        R.append(r); S.append(s); F.append(f)
    return float(np.mean(R)), float(np.mean(S)), float(np.mean(F))
