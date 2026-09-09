"""E3 (tabular, exact evaluation): the adaptation step size alpha as a
personalization knob.

Chunked by variant (each neural-free run is a few seconds but there are many):
  python e3_alpha.py exact|fo|fedavg   # one variant over all alphas/seeds
  python e3_alpha.py merge             # combine -> results/e3.npz
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from per_fedavg_pg import gridworld as G

CFG = dict(W=5, H=15, gamma=0.9, beta=0.3, tau=5, K=80,
           m_in=10, m_h=10, m_out=10, seeds=list(range(10)),
           alphas=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def run_variant(v):
    mdps = G.make_agents(W=CFG['W'], H=CFG['H'], gamma=CFG['gamma']); n = len(mdps)
    Fm, Fs_, Gm, Gs_ = [], [], [], []
    for a in CFG['alphas']:
        Fs, Gs = [], []
        for s in CFG['seeds']:
            r = G.train(mdps, v, K=CFG['K'], tau=CFG['tau'], alpha=a, beta=CFG['beta'],
                        m_in=CFG['m_in'], m_h=CFG['m_h'], m_out=CFG['m_out'],
                        r=n, seed=s, eval_every=CFG['K'])
            Fs.append(r['Fpost'][-1]); Gs.append(r['Gnorm'][-1])
        Fm.append(np.mean(Fs)); Fs_.append(np.std(Fs)); Gm.append(np.mean(Gs)); Gs_.append(np.std(Gs))
    return dict(F_mean=np.array(Fm), F_std=np.array(Fs_), G_mean=np.array(Gm), G_std=np.array(Gs_))


def main():
    arg = sys.argv[1]
    if arg == 'merge':
        data = {'config': json.dumps(CFG), 'alphas': np.array(CFG['alphas'])}
        for v in ['exact', 'fo', 'fedavg']:
            d = dict(np.load(os.path.join(RESDIR, f'e3_{v}.npz')))
            for k, val in d.items():
                data[f'{v}/{k}'] = val
        np.savez(os.path.join(RESDIR, 'e3.npz'), **data)
        print("merged -> results/e3.npz")
    else:
        d = run_variant(arg)
        np.savez(os.path.join(RESDIR, f'e3_{arg}.npz'), **d)
        print(f"{arg}: F by alpha = " +
              " ".join(f"a{a}:{m:.2f}" for a, m in zip(CFG['alphas'], d['F_mean'])))


if __name__ == '__main__':
    main()
