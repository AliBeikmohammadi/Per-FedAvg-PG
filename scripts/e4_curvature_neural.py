"""E4 (neural, Monte-Carlo evaluation): the curvature-estimation bottleneck.

Usage:
  python e4_curvature_neural.py <m_h>     # run one m_h over all seeds -> partial
  python e4_curvature_neural.py fedavg    # FedAvg reference
  python e4_curvature_neural.py merge     # combine partials -> results/e4.npz

Chunked by m_h because each neural run takes tens of seconds.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from per_fedavg_pg import neural as N
from per_fedavg_pg.tasks import make_arc_family

CFG = dict(radius=0.8, n_train=6, n_held=3, span=[-0.5, 0.5],
           alpha=1.0, beta=0.2, tau=5, K=150, m_in=10, m_out=10,
           m_h_grid=[10, 50, 100, 200, 500, 1000], seeds=list(range(10)),
           m_adapt_eval=256, m_eval=512)
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def run_config(variant, m_h):
    train_goals, _, _ = make_arc_family(CFG['radius'], CFG['n_train'],
                                        CFG['n_held'], tuple(CFG['span']))
    vals = []
    for s in CFG['seeds']:
        r = N.train(train_goals, variant, K=CFG['K'], tau=CFG['tau'],
                    alpha=CFG['alpha'], beta=CFG['beta'], m_in=CFG['m_in'],
                    m_h=m_h, m_out=CFG['m_out'], r=CFG['n_train'], seed=s,
                    eval_every=CFG['K'])
        # exact post-adaptation value on the TRAINING agents (MC, fixed seed)
        F, _ = N.eval_mc(train_goals, r['theta'], CFG['alpha'],
                         seed=123456, m_adapt=CFG['m_adapt_eval'], m_eval=CFG['m_eval'])
        vals.append(F)
    return np.array(vals)


def main():
    arg = sys.argv[1]
    if arg == 'merge':
        data = {'config': json.dumps(CFG)}
        mhs, means, stds = [], [], []
        for m_h in CFG['m_h_grid']:
            f = os.path.join(RESDIR, f'e4_mh{m_h}.npz')
            if os.path.exists(f):
                v = np.load(f)['vals']
                mhs.append(m_h); means.append(v.mean()); stds.append(v.std())
                data[f'mh{m_h}/vals'] = v
        data['m_h'] = np.array(mhs)
        data['exact_mean'] = np.array(means)
        data['exact_std'] = np.array(stds)
        fref = os.path.join(RESDIR, 'e4_fedavg.npz')
        if os.path.exists(fref):
            fv = np.load(fref)['vals']
            data['fedavg_mean'] = np.array([fv.mean()])
            data['fedavg_std'] = np.array([fv.std()])
        np.savez(os.path.join(RESDIR, 'e4.npz'), **data)
        print("merged -> results/e4.npz  m_h:", mhs, "exact means:",
              [round(x, 3) for x in means])
    elif arg == 'fedavg':
        v = run_config('fedavg', 10)  # fedavg is m_h-independent
        np.savez(os.path.join(RESDIR, 'e4_fedavg.npz'), vals=v)
        print(f"fedavg ref: F={v.mean():.3f}+-{v.std():.3f}")
    else:
        m_h = int(arg)
        v = run_config('exact', m_h)
        np.savez(os.path.join(RESDIR, f'e4_mh{m_h}.npz'), vals=v)
        print(f"exact m_h={m_h}: F={v.mean():.3f}+-{v.std():.3f}  vals={np.round(v,3)}")


if __name__ == '__main__':
    main()
