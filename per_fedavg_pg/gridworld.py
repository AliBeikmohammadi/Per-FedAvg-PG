"""
Library for the numerical experiments of
"Personalized Federated Reinforcement Learning via MAML" (working draft).

Environment: W x W gridworld, 4 actions (up/down/left/right), deterministic
moves (walls: stay in place), per-agent goal cell, reward r_i(s) = 1{s == goal_i}
collected at every time step, finite horizon H, discount gamma < 1, uniform
initial distribution over non-goal cells.

Policy: shared tabular softmax, theta in R^{S x A},
pi(a|s;theta) = softmax(theta[s,:])[a]. 

Estimators:
    g(tau;theta)      = sum_h grad log pi(a_h|s_h) * R^h,   R^h = sum_{t>=h} gamma^t r_t
    u(tau;theta)      = g * (grad log q)^T + Hess(nu)
    meta-grad         = (I + alpha * H_hat) g_out          (three independent batches)

Exact evaluation: finite-horizon dynamic programming gives J_i(theta) and
grad J_i(theta) exactly, hence the exact deterministic-adaptation objective
    F(theta) = (1/n) sum_i J_i(theta + alpha * grad J_i(theta))
used for all reported curves (no evaluation noise).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

class GridMDP:
    def __init__(self, W, goal, H, gamma):
        self.W = W
        self.S = W * W
        self.A = 4
        self.goal = goal
        self.H = H
        self.gamma = gamma
        nxt = np.zeros((self.S, 4), dtype=np.int64)
        for s in range(self.S):
            x, y = s % W, s // W
            moves = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]  # U, D, L, R
            for a, (nx, ny) in enumerate(moves):
                nxt[s, a] = (ny * W + nx) if (0 <= nx < W and 0 <= ny < W) else s
        self.nxt = nxt
        self.r = np.zeros(self.S)
        self.r[goal] = 1.0
        mu = np.ones(self.S)
        mu[goal] = 0.0
        self.mu = mu / mu.sum()
        self.disc = gamma ** np.arange(H + 1)          # gamma^t, t = 0..H


def softmax(theta):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Vectorized rollouts and per-trajectory statistics
# ----------------------------------------------------------------------------

def rollout(mdp, pi, m, rng):
    """m trajectories under policy pi. Returns states, actions (m, H+1)."""
    Hp1 = mdp.H + 1
    states = np.empty((m, Hp1), dtype=np.int64)
    actions = np.empty((m, Hp1), dtype=np.int64)
    s = rng.choice(mdp.S, size=m, p=mdp.mu)
    u = rng.random((m, Hp1))
    for h in range(Hp1):
        states[:, h] = s
        cdf = np.cumsum(pi[s], axis=1)
        a = (u[:, h][:, None] > cdf).sum(axis=1)
        np.clip(a, 0, mdp.A - 1, out=a)
        actions[:, h] = a
        s = mdp.nxt[s, a]
    return states, actions


def reward_to_go(mdp, states):
    """R^h = sum_{t=h}^H gamma^t r(s_t), shape (m, H+1)."""
    dr = mdp.r[states] * mdp.disc[None, :]
    return np.cumsum(dr[:, ::-1], axis=1)[:, ::-1]


def traj_stats(mdp, theta, m, rng, need_hess=False):
    """Per-trajectory statistics at theta.
    Returns g (m,S,A) = policy gradients, and if need_hess:
    sc (m,S,A) = grad log q, c (m,S) = per-state sums of R^h (for Hess(nu))."""
    pi = softmax(theta)
    states, actions = rollout(mdp, pi, m, rng)
    R = reward_to_go(mdp, states)
    idx = np.arange(m)
    g = np.zeros((m, mdp.S, mdp.A))
    sc = np.zeros((m, mdp.S, mdp.A)) if need_hess else None
    c = np.zeros((m, mdp.S)) if need_hess else None
    for h in range(mdp.H + 1):
        s_h, a_h, w = states[:, h], actions[:, h], R[:, h]
        p = pi[s_h]                                   # (m, A)
        g[idx, s_h, :] -= w[:, None] * p
        g[idx, s_h, a_h] += w
        if need_hess:
            sc[idx, s_h, :] -= p
            sc[idx, s_h, a_h] += 1.0
            c[idx, s_h] += w
    return (g, sc, c, pi) if need_hess else (g, None, None, pi)


def pg_estimate(mdp, theta, m, rng):
    g, _, _, _ = traj_stats(mdp, theta, m, rng, need_hess=False)
    return g.mean(axis=0)


def hvp_estimate(mdp, theta, v, m, rng):
    """Monte Carlo estimate of Hess J(theta) @ v via u(tau;theta) v =
    g (sc . v) + Hess(nu) v, averaged over m fresh trajectories."""
    g, sc, c, pi = traj_stats(mdp, theta, m, rng, need_hess=True)
    dots = np.einsum('msa,sa->m', sc, v)              # (m,)
    term1 = (g * dots[:, None, None]).mean(axis=0)    # (S,A)
    # Hess(nu) v: block s of Hess(nu) is -c_s (diag(pi_s) - pi_s pi_s^T)
    pv = pi * v                                        # (S,A)
    pvsum = pv.sum(axis=1, keepdims=True)              # (S,1)
    block = -(pv - pi * pvsum)                         # (S,A) = -(diag(pi)-pi pi^T) v per state
    term2 = (c.mean(axis=0))[:, None] * block          # (S,A)
    return term1 + term2


def meta_grad(mdp, theta, alpha, m_in, m_h, m_out, rng, variant):
    """One stochastic meta-gradient.
    variant in {'exact','fo','fedavg'}. 'fedavg' = plain PG with the same
    total trajectory budget."""
    if variant == 'fedavg':
        return pg_estimate(mdp, theta, m_in + m_h + m_out, rng)
    g_in = pg_estimate(mdp, theta, m_in, rng)
    th_ad = theta + alpha * g_in
    g_out = pg_estimate(mdp, th_ad, m_out, rng)
    if variant == 'fo':
        return g_out
    Hv = hvp_estimate(mdp, theta, g_out, m_h, rng)     # independent batch at theta
    return g_out + alpha * Hv


# ----------------------------------------------------------------------------
# Exact evaluation via finite-horizon dynamic programming
# ----------------------------------------------------------------------------

def exact_J_grad(mdp, theta):
    """Exact J(theta) and grad J(theta) for the finite-horizon discounted
    objective J = E[ sum_{t=0}^H gamma^t r(s_t) ]."""
    pi = softmax(theta)
    S, A, H = mdp.S, mdp.A, mdp.H
    # forward: state distributions p_h
    p = np.zeros((H + 1, S))
    p[0] = mdp.mu
    for h in range(H):
        contrib = p[h][:, None] * pi                  # (S,A)
        pn = np.zeros(S)
        np.add.at(pn, mdp.nxt.ravel(), contrib.ravel())
        p[h + 1] = pn
    # backward: Q_h(s,a) = E[R^h | s_h=s, a_h=a] with absolute discounting
    V = np.zeros((H + 2, S))
    grad = np.zeros((S, A))
    J = None
    Qs = [None] * (H + 1)
    for h in range(H, -1, -1):
        Q = (mdp.gamma ** h) * mdp.r[:, None] + V[h + 1][mdp.nxt]
        V[h] = (pi * Q).sum(axis=1)
        Qs[h] = Q
    J = float(mdp.mu @ V[0])
    for h in range(H + 1):
        adv = Qs[h] - V[h][:, None]
        grad += p[h][:, None] * pi * adv
    return J, grad


def _fd_hvp(mdp, theta, v, eps=1e-5):
    """Exact Hessian-vector product Hess J(theta) @ v via central finite
    differences of the exact gradient."""
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.zeros_like(v)
    u = v / nv
    _, gp = exact_J_grad(mdp, theta + eps * u)
    _, gm = exact_J_grad(mdp, theta - eps * u)
    return nv * (gp - gm) / (2 * eps)


def eval_exact(mdps, theta, alpha_eval, with_grad=False):
    """Exact pre-adaptation f(theta), post-adaptation F(theta), and (optionally)
    the exact squared meta-gradient norm ||grad F(theta)||^2, averaged over agents.
    grad F_i(theta) = (I + alpha Hess J_i(theta)) grad J_i(theta + alpha grad J_i(theta))."""
    Fs, fs = [], []
    G = np.zeros_like(theta) if with_grad else None
    for mdp in mdps:
        J0, g0 = exact_J_grad(mdp, theta)
        th_ad = theta + alpha_eval * g0
        J1, g1 = exact_J_grad(mdp, th_ad)
        fs.append(J0)
        Fs.append(J1)
        if with_grad:
            G += g1 + alpha_eval * _fd_hvp(mdp, theta, g1)
    if with_grad:
        G /= len(mdps)
        return float(np.mean(Fs)), float(np.mean(fs)), float(np.sum(G * G))
    return float(np.mean(Fs)), float(np.mean(fs))


# ----------------------------------------------------------------------------
# Per-FedAvg-PG training loop (Algorithm 2 of the paper), with client sampling
# ----------------------------------------------------------------------------

def train(mdps, variant, K, tau, alpha, beta, m_in, m_h, m_out, r, seed,
          eval_every=5, alpha_eval=None):
    rng = np.random.default_rng(seed)
    n = len(mdps)
    if alpha_eval is None:
        alpha_eval = alpha
    theta = np.zeros((mdps[0].S, mdps[0].A))
    rounds, Fpost, fpre, Gnorm = [], [], [], []
    for k in range(K + 1):
        if k % eval_every == 0 or k == K:
            F, f, gn = eval_exact(mdps, theta, alpha_eval, with_grad=True)
            rounds.append(k)
            Fpost.append(F)
            fpre.append(f)
            Gnorm.append(gn)
        if k == K:
            break
        cohort = rng.choice(n, size=r, replace=False)
        acc = np.zeros_like(theta)
        for i in cohort:
            th = theta.copy()
            for _ in range(tau):
                th = th + beta * meta_grad(mdps[i], th, alpha,
                                           m_in, m_h, m_out, rng, variant)
            acc += th
        theta = acc / r
    return dict(rounds=np.array(rounds), Fpost=np.array(Fpost),
                fpre=np.array(fpre), Gnorm=np.array(Gnorm), theta=theta)


def make_agents(W=5, H=15, gamma=0.9):
    """n = 8 agents: goals at the 4 corners and 4 edge midpoints."""
    W2 = W // 2
    cells = [(0, 0), (W - 1, 0), (0, W - 1), (W - 1, W - 1),
             (W2, 0), (0, W2), (W - 1, W2), (W2, W - 1)]
    return [GridMDP(W, y * W + x, H, gamma) for (x, y) in cells]


# ----------------------------------------------------------------------------
# Sanity checks: MC estimators vs exact quantities
# ----------------------------------------------------------------------------

def sanity_checks(seed=0):
    rng = np.random.default_rng(seed)
    mdp = make_agents()[0]
    theta = rng.normal(scale=0.5, size=(mdp.S, mdp.A))

    # 1) exact gradient vs finite differences of exact J
    J, g = exact_J_grad(mdp, theta)
    eps = 1e-5
    errs = []
    for _ in range(12):
        s, a = rng.integers(mdp.S), rng.integers(mdp.A)
        tp = theta.copy(); tp[s, a] += eps
        tm = theta.copy(); tm[s, a] -= eps
        fd = (exact_J_grad(mdp, tp)[0] - exact_J_grad(mdp, tm)[0]) / (2 * eps)
        errs.append(abs(fd - g[s, a]) / (abs(fd) + 1e-12))
    print(f"[1] exact grad vs finite diff of exact J: max rel err = {max(errs):.2e}")

    # 2) MC policy gradient vs exact gradient
    ghat = pg_estimate(mdp, theta, 200000, rng)
    rel = np.linalg.norm(ghat - g) / np.linalg.norm(g)
    print(f"[2] MC grad (m=2e5) vs exact grad:      rel L2 err  = {rel:.3f}")

    # 3) MC Hessian-vector product vs finite difference of exact gradients
    v = rng.normal(size=(mdp.S, mdp.A)); v /= np.linalg.norm(v)
    hv_hat = hvp_estimate(mdp, theta, v, 200000, rng)
    e = 1e-4
    hv_fd = (exact_J_grad(mdp, theta + e * v)[1]
             - exact_J_grad(mdp, theta - e * v)[1]) / (2 * e)
    rel = np.linalg.norm(hv_hat - hv_fd) / np.linalg.norm(hv_fd)
    print(f"[3] MC HVP (m=2e5) vs finite-diff HVP:  rel L2 err  = {rel:.3f}")


if __name__ == "__main__":
    sanity_checks()
