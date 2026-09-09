"""E5 (neural, Monte-Carlo evaluation): few-shot adaptation to HELD-OUT agents
-- the headline experiment.

Meta-train a shared initialization with the FIRST-ORDER variant on the training
agents of the shared-structure arc family, then measure performance on the
HELD-OUT (unseen) agents as a function of the number of adaptation trajectories.
The analyzed objective adapts with the *deterministic* gradient, so we adapt
with ONE policy-gradient step and vary the adaptation batch m_adapt (the number
of trajectories used to form that step); m_adapt=0 is the zero-shot init.
Baselines: from-scratch (independent) training of each held-out agent and its
large-budget specialist limit. Three readouts of the same rollouts are recorded:
expected return, success rate (fraction of episodes reaching the goal region),
and mean final distance.

Output: results/e5.npz
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from per_fedavg_pg import neural as N
from per_fedavg_pg.tasks import (make_arc_family, eval_policy,
                                   few_shot_metrics_from_init, from_scratch_metrics)

CFG = dict(radius=0.8, n_train=6, n_held=3, span=[-0.5, 0.5],
           alpha=1.0, beta=0.2, tau=5, K=150, m_in=10, m_h=10, m_out=10,
           meta_variant='fo', seeds=list(range(10)),
           meta_madapt=[0, 20, 50, 100, 200, 500],   # one step; 0 = zero-shot
           scratch_m=20, scratch_G=[1, 5, 25, 100, 200], m_eval=512)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'e5.npz')


def main():
    train_goals, held_goals, info = make_arc_family(
        CFG['radius'], CFG['n_train'], CFG['n_held'], tuple(CFG['span']))
    S, KM, KS = len(CFG['seeds']), len(CFG['meta_madapt']), len(CFG['scratch_G'])
    meta_ret, meta_suc, meta_fd = (np.zeros((S, KM)) for _ in range(3))
    scr_ret, scr_suc, scr_fd = (np.zeros((S, KS)) for _ in range(3))

    for si, s in enumerate(CFG['seeds']):
        theta_star = N.train(train_goals, CFG['meta_variant'], K=CFG['K'],
                             tau=CFG['tau'], alpha=CFG['alpha'], beta=CFG['beta'],
                             m_in=CFG['m_in'], m_h=CFG['m_h'], m_out=CFG['m_out'],
                             r=CFG['n_train'], seed=s, eval_every=CFG['K'])['theta']
        for ki, ma in enumerate(CFG['meta_madapt']):
            if ma == 0:                       # zero-shot
                rs = [eval_policy(g, theta_star, CFG['m_eval'], seed=30_000 + i)
                      for i, g in enumerate(held_goals)]
                r, sc, fd = (float(np.mean([x[j] for x in rs])) for j in range(3))
            else:                             # one adaptation step with batch ma
                r, sc, fd = few_shot_metrics_from_init(
                    held_goals, theta_star, CFG['alpha'], ma, 1,
                    m_eval=CFG['m_eval'], seed=7 + s)
            meta_ret[si, ki], meta_suc[si, ki], meta_fd[si, ki] = r, sc, fd
        for gi, G in enumerate(CFG['scratch_G']):
            r, sc, fd = from_scratch_metrics(held_goals, G, CFG['scratch_m'],
                                             CFG['beta'], m_eval=CFG['m_eval'],
                                             seed=7 + s, init_seed=100 + s)
            scr_ret[si, gi], scr_suc[si, gi], scr_fd[si, gi] = r, sc, fd
        print(f"seed {s}: zero-shot succ={meta_suc[si,0]:.2f}  1-step@100 succ="
              f"{meta_suc[si, CFG['meta_madapt'].index(100)]:.2f}  scratch@2000traj succ={scr_suc[si,3]:.2f}", flush=True)

    rnd = [eval_policy(g, N.init_theta(seed=100 + s), CFG['m_eval'], seed=40_000 + i)
           for s in CFG['seeds'] for i, g in enumerate(held_goals)]
    np.savez(OUT, config=json.dumps({**CFG, 'ang_held': info['ang_held'].tolist()}),
             meta_traj=np.array(CFG['meta_madapt']),
             scratch_traj=np.array(CFG['scratch_G']) * CFG['scratch_m'],
             meta_curves=meta_ret, meta_success=meta_suc, meta_finaldist=meta_fd,
             scratch_curves=scr_ret, scratch_success=scr_suc, scratch_finaldist=scr_fd,
             random_ref=np.array([np.mean([x[0] for x in rnd])]),
             random_success=np.array([np.mean([x[1] for x in rnd])]))
    i100 = CFG['meta_madapt'].index(100)
    print(f"\nHELD-OUT (mean/{S} seeds): zero-shot succ={meta_suc[:,0].mean():.0%} return={meta_ret[:,0].mean():.2f}; "
          f"1-step@100 succ={meta_suc[:,i100].mean():.0%} return={meta_ret[:,i100].mean():.2f}; "
          f"scratch@2000traj succ={scr_suc[:,3].mean():.0%}; scratch@4000 succ={scr_suc[:,-1].mean():.0%}")
    print("saved", OUT)


if __name__ == '__main__':
    main()
