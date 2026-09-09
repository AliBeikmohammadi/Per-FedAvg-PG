"""E1 + E2 (tabular, exact dynamic-programming evaluation).

E1 -- personalization within federation: exact / FO / FedAvg all reach a shared
      initialization; evaluated by the exact post-adaptation objective F.
E2 -- rates & the FO floor: exact ||grad F(theta^k)||^2 vs rounds

Both come from the same training runs (train() records Fpost and Gnorm), so a
single script produces the data for both figures. Tabular is used precisely
because finite MDPs give F and grad F exactly (no evaluation noise).

Output: results/e1e2.npz
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from per_fedavg_pg import gridworld as G
from per_fedavg_pg.tasks import per_step_cost

CFG = dict(W=5, H=15, gamma=0.9, alpha=2.0, beta=0.3, tau=5, K=80,
           m_in=10, m_h=10, m_out=10, eval_every=5, seeds=list(range(10)))
VARIANTS = ['exact', 'fo', 'fedavg']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'e1e2.npz')


def main():
    mdps = G.make_agents(W=CFG['W'], H=CFG['H'], gamma=CFG['gamma'])
    n = len(mdps)
    data = {'config': json.dumps({**CFG, 'n': n, 'variants': VARIANTS})}
    # random-policy floor: value of the uniform policy (theta = 0 -> uniform softmax),
    # evaluated exactly (pre-adaptation) and averaged over agents.
    _, f_rand = G.eval_exact(mdps, np.zeros((mdps[0].S, mdps[0].A)), CFG['alpha'])
    data['random_floor'] = np.array([f_rand])
    for v in VARIANTS:
        cost = per_step_cost(v, CFG['m_in'], CFG['m_h'], CFG['m_out'])
        Fp, fp, Gn = [], [], []
        for s in CFG['seeds']:
            r = G.train(mdps, v, K=CFG['K'], tau=CFG['tau'], alpha=CFG['alpha'],
                        beta=CFG['beta'], m_in=CFG['m_in'], m_h=CFG['m_h'],
                        m_out=CFG['m_out'], r=n, seed=s, eval_every=CFG['eval_every'])
            Fp.append(r['Fpost']); fp.append(r['fpre']); Gn.append(r['Gnorm']); rounds = r['rounds']
        Fp, fp, Gn = np.array(Fp), np.array(fp), np.array(Gn)   # (seeds, checkpoints)
        data[f'{v}/rounds'] = rounds
        data[f'{v}/per_agent_traj'] = rounds * CFG['tau'] * cost
        data[f'{v}/Fpost'] = Fp        # post-adaptation value F(theta^k)
        data[f'{v}/fpre'] = fp         # pre-adaptation value (shared init, before the alpha-step)
        data[f'{v}/Gnorm'] = Gn
        print(f"{v:7s}: F_post={Fp[:,-1].mean():.3f}  f_pre={fp[:,-1].mean():.3f}  "
              f"||gradF||^2={Gn[:,-1].mean():.4f}  (cost/step={cost} traj)")
    print(f"random-policy floor: {f_rand:.3f}")
    np.savez(OUT, **data)
    print("saved", OUT)


if __name__ == '__main__':
    main()
