"""E7 (neural): trajectory visualization on a HELD-OUT agent.

Rolls out the first-order meta-initialization on an unseen goal BEFORE and
AFTER a single adaptation step, and a from-scratch policy at the same small
adaptation budget, saving the 2-D paths for plotting (make_figures: fig_e7).
Self-contained (trains its own meta-init, ~15-30 s). Output: results/e7_traj.npz
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from per_fedavg_pg import neural as N
from per_fedavg_pg.tasks import make_arc_family, adapt_from_init

CFG = dict(radius=0.8, n_train=6, n_held=3, span=[-0.5, 0.5], alpha=1.0, beta=0.2,
           tau=5, K=150, m_in=10, m_h=10, m_out=10, seed=0, m_adapt=200,
           n_paths=12, held_index=0, scratch_steps=10)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'e7_traj.npz')


def paths(goal, theta, m, seed):
    states, _, _ = N.rollout(goal, theta, m, np.random.default_rng(seed))
    return states                      # (m, H+1, 2)


def main():
    train_goals, held_goals, _ = make_arc_family(
        CFG['radius'], CFG['n_train'], CFG['n_held'], tuple(CFG['span']))
    goal = held_goals[CFG['held_index']]
    theta_star = N.train(train_goals, 'fo', K=CFG['K'], tau=CFG['tau'], alpha=CFG['alpha'],
                         beta=CFG['beta'], m_in=CFG['m_in'], m_h=CFG['m_h'],
                         m_out=CFG['m_out'], r=CFG['n_train'], seed=CFG['seed'],
                         eval_every=CFG['K'])['theta']
    rng = np.random.default_rng(123)
    theta_adapt = adapt_from_init(goal, theta_star, CFG['alpha'], CFG['m_adapt'], 1, rng)
    theta_scratch = N.init_theta(seed=100)
    sb = CFG['m_adapt'] // CFG['scratch_steps']    # per-step batch so total == m_adapt (200 traj)
    for _ in range(CFG['scratch_steps']):          # from-scratch at the SAME total budget
        theta_scratch = theta_scratch + CFG['beta'] * N.pg_estimate(goal, theta_scratch, sb, rng)
    n = CFG['n_paths']
    np.savez(OUT, goal=goal, goal_r=N.GOAL_R,
             before=paths(goal, theta_star, n, 1),
             after=paths(goal, theta_adapt, n, 2),
             scratch=paths(goal, theta_scratch, n, 3),
             config=json.dumps(CFG))
    print("saved", OUT)


if __name__ == '__main__':
    main()
