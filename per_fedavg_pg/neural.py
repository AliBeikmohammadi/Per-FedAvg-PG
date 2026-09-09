"""Neural-policy experiment: 2D continuous navigation with an MLP softmax policy.

Environment: state p in [-1,1]^2, 8 unit-direction actions with step 0.15
(deterministic, clipped to the box), horizon H=20, discount 0.9, reward
r_i(p) = 1{ ||p - goal_i|| <= 0.2 } collected each step; n=8 agents with
goals equally spaced on the circle of radius 0.8; uniform initial state.

Policy: 2 -> 32 tanh -> 8 softmax MLP (D = 360 parameters), shared across
agents. The score Hessian term of u is computed by central finite differences of 
the (weighted) score gradient (exact up to O(eps^2) in float64), so the 'exact' 
variant is implementable without second-order autodiff.

Evaluation is Monte Carlo (no DP in continuous state): adaptation uses a
large batch (m=256) approximating the deterministic-adaptation objective,
and returns are averaged over 512 rollouts per agent.
"""

import numpy as np

H, GAMMA, STEP, GOAL_R = 20, 0.9, 0.15, 0.2
NA, NH = 8, 32                      # actions, hidden units
D = 2 * NH + NH + NH * NA + NA      # 360
DIRS = np.stack([np.array([np.cos(t), np.sin(t)])
                 for t in 2 * np.pi * np.arange(NA) / NA])
DISC = GAMMA ** np.arange(H + 1)


def init_theta(seed=123):
    """Fixed shared initialization: random features so gradients reach all
    layers (all-zero init freezes W1, b1, W2 -- only b2 would train)."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(scale=0.5, size=(2, NH))
    W2 = rng.normal(scale=0.1, size=(NH, NA))
    return np.concatenate([W1.ravel(), np.zeros(NH), W2.ravel(), np.zeros(NA)])


def make_nav_agents(n=8, radius=0.8):
    return [radius * np.array([np.cos(t), np.sin(t)])
            for t in 2 * np.pi * (np.arange(n) + 0.5) / n]


def unpack(theta):
    i = 0
    W1 = theta[i:i + 2 * NH].reshape(2, NH); i += 2 * NH
    b1 = theta[i:i + NH]; i += NH
    W2 = theta[i:i + NH * NA].reshape(NH, NA); i += NH * NA
    b2 = theta[i:i + NA]
    return W1, b1, W2, b2


def forward(theta, x):
    """x (N,2) -> pi (N,NA), cache for backprop."""
    W1, b1, W2, b2 = unpack(theta)
    h = np.tanh(x @ W1 + b1)
    logits = h @ W2 + b2
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    pi = e / e.sum(axis=1, keepdims=True)
    return pi, h


def weighted_score_grads(theta, states, actions, w, per_traj=True):
    """Gradients of sum_h w[:,h] * log pi(a_h | s_h) w.r.t. theta.
    states (m,T,2), actions (m,T), w (m,T). Returns (m,D) if per_traj else (D,)."""
    W1, b1, W2, b2 = unpack(theta)
    m, T = actions.shape
    x = states.reshape(m * T, 2)
    pi, h = forward(theta, x)
    dlogits = -pi
    dlogits[np.arange(m * T), actions.ravel()] += 1.0
    dlogits *= w.reshape(m * T, 1)
    dh = dlogits @ W2.T
    dz1 = dh * (1.0 - h * h)
    if per_traj:
        g = np.empty((m, D))
        dW1 = (x[:, :, None] * dz1[:, None, :]).reshape(m, T, 2 * NH).sum(1)
        db1 = dz1.reshape(m, T, NH).sum(1)
        dW2 = (h[:, :, None] * dlogits[:, None, :]).reshape(m, T, NH * NA).sum(1)
        db2 = dlogits.reshape(m, T, NA).sum(1)
        g[:, :2 * NH] = dW1
        g[:, 2 * NH:2 * NH + NH] = db1
        g[:, 2 * NH + NH:2 * NH + NH + NH * NA] = dW2
        g[:, -NA:] = db2
        return g
    out = np.empty(D)
    out[:2 * NH] = (x.T @ dz1).ravel()
    out[2 * NH:2 * NH + NH] = dz1.sum(0)
    out[2 * NH + NH:2 * NH + NH + NH * NA] = (h.T @ dlogits).ravel()
    out[-NA:] = dlogits.sum(0)
    return out


def rollout(goal, theta, m, rng):
    states = np.empty((m, H + 1, 2))
    actions = np.empty((m, H + 1), dtype=np.int64)
    p = rng.uniform(-1.0, 1.0, (m, 2))
    u = rng.random((m, H + 1))
    for h_ in range(H + 1):
        states[:, h_] = p
        pi, _ = forward(theta, p)
        cdf = np.cumsum(pi, axis=1)
        a = (u[:, h_][:, None] > cdf).sum(axis=1)
        np.clip(a, 0, NA - 1, out=a)
        actions[:, h_] = a
        p = np.clip(p + STEP * DIRS[a], -1.0, 1.0)
    dist = np.linalg.norm(states - goal[None, None, :], axis=2)
    rew = (dist <= GOAL_R).astype(float)
    R = np.cumsum((rew * DISC[None, :])[:, ::-1], axis=1)[:, ::-1]
    return states, actions, R


def pg_estimate(goal, theta, m, rng):
    states, actions, R = rollout(goal, theta, m, rng)
    return weighted_score_grads(theta, states, actions, R, per_traj=False) / m


def hvp_estimate(goal, theta, v, m, rng, eps=1e-4):
    """(1/m) sum_j u(xi_j; theta) v via per-trajectory outer parts + batched FD
    of the weighted score gradient for the Hessian-of-nu part."""
    states, actions, R = rollout(goal, theta, m, rng)
    g = weighted_score_grads(theta, states, actions, R, per_traj=True)      # (m,D)
    sc = weighted_score_grads(theta, states, actions, np.ones_like(R), True)
    term1 = (g * (sc @ v)[:, None]).mean(axis=0)
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return term1
    u_ = v / nv
    gp = weighted_score_grads(theta + eps * u_, states, actions, R, False)
    gm = weighted_score_grads(theta - eps * u_, states, actions, R, False)
    term2 = nv * (gp - gm) / (2 * eps * m)
    return term1 + term2


def meta_grad(goal, theta, alpha, m_in, m_h, m_out, rng, variant):
    if variant == 'fedavg':
        return pg_estimate(goal, theta, m_in + m_h + m_out, rng)
    g_in = pg_estimate(goal, theta, m_in, rng)
    th_ad = theta + alpha * g_in
    g_out = pg_estimate(goal, th_ad, m_out, rng)
    if variant == 'fo':
        return g_out
    return g_out + alpha * hvp_estimate(goal, theta, g_out, m_h, rng)


def eval_mc(goals, theta, alpha_eval, seed=123456, m_adapt=256, m_eval=512):
    """Monte Carlo estimate of pre-/post-adaptation performance (fixed eval seed)."""
    rng = np.random.default_rng(seed)
    Fs, fs = [], []
    for goal in goals:
        _, _, R0 = rollout(goal, theta, m_eval, rng)
        fs.append(R0[:, 0].mean())
        g = pg_estimate(goal, theta, m_adapt, rng)
        _, _, R1 = rollout(goal, theta + alpha_eval * g, m_eval, rng)
        Fs.append(R1[:, 0].mean())
    return float(np.mean(Fs)), float(np.mean(fs))


def train(goals, variant, K, tau, alpha, beta, m_in, m_h, m_out, r, seed,
          eval_every=10, alpha_eval=None):
    rng = np.random.default_rng(seed)
    n = len(goals)
    if alpha_eval is None:
        alpha_eval = alpha
    theta = init_theta()
    rounds, Fpost, fpre = [], [], []
    for k in range(K + 1):
        if k % eval_every == 0 or k == K:
            F, f = eval_mc(goals, theta, alpha_eval)
            rounds.append(k); Fpost.append(F); fpre.append(f)
        if k == K:
            break
        cohort = rng.choice(n, size=r, replace=False)
        acc = np.zeros_like(theta)
        for i in cohort:
            th = theta.copy()
            for _ in range(tau):
                th = th + beta * meta_grad(goals[i], th, alpha,
                                           m_in, m_h, m_out, rng, variant)
            acc += th
        theta = acc / r
    return dict(rounds=np.array(rounds), Fpost=np.array(Fpost),
                fpre=np.array(fpre), theta=theta)


def sanity(seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=0.3, size=D)
    goal = make_nav_agents()[0]
    # 1) analytic weighted score gradient vs finite differences
    # (strictly positive weights so the check cannot be vacuously 0=0)
    states, actions, R = rollout(goal, theta, 5, rng)
    R = rng.uniform(0.5, 1.5, size=R.shape)
    g = weighted_score_grads(theta, states, actions, R, per_traj=False)

    def obj(th):
        m, T = actions.shape
        pi, _ = forward(th, states.reshape(m * T, 2))
        lp = np.log(pi[np.arange(m * T), actions.ravel()] + 1e-300)
        return float((lp * R.ravel()).sum())
    eps, errs = 1e-6, []
    for j in rng.integers(0, D, 12):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        fd = (obj(tp) - obj(tm)) / (2 * eps)
        errs.append(abs(fd - g[j]) / (abs(fd) + 1e-12))
    print(f"[1] analytic score grad vs FD: max rel err = {max(errs):.2e}")
    # 2) hvp self-consistency: batched FD vs per-trajectory FD at smaller eps
    v = rng.normal(size=D); v /= np.linalg.norm(v)
    h1 = hvp_estimate(goal, theta, v, 4000, np.random.default_rng(1), eps=1e-4)
    h2 = hvp_estimate(goal, theta, v, 4000, np.random.default_rng(1), eps=1e-5)
    print(f"[2] HVP eps-consistency: rel diff = "
          f"{np.linalg.norm(h1-h2)/np.linalg.norm(h2):.2e}")


if __name__ == '__main__':
    sanity()


def rollout_metrics(goal, theta, m, rng):
    """Interpretable readouts on m fresh episodes:
      success rate  = fraction of episodes that reach within GOAL_R of the goal
                      at some step (i.e. the goal region is visited);
      final_dist    = mean distance to the goal at the last step.
    """
    states, _, _ = rollout(goal, theta, m, rng)
    dist = np.linalg.norm(states - goal[None, None, :], axis=2)   # (m, H+1)
    success = float((dist <= GOAL_R).any(axis=1).mean())
    final_dist = float(dist[:, -1].mean())
    return success, final_dist
