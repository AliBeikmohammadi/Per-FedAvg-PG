"""Correctness checks for the experiment suite. Run: python run_tests.py

Covers: (1) gridworld library sanity (exact grad vs finite differences, exact-DP
vs MC), (2) neural library sanity, (3) trajectory-accounting formulas,
(4) determinism (same seed -> identical results) for both environments.
Exit code 0 iff all pass.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_fedavg_pg import gridworld as G
from per_fedavg_pg import neural as N
from per_fedavg_pg.tasks import per_step_cost, make_arc_family

ok = True
def check(name, cond):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    ok = ok and bool(cond)


def main():
    print("[1] gridworld library sanity")
    try:
        G.sanity_checks(seed=0); check("gridworld.sanity_checks ran", True)
    except AssertionError as e:
        check(f"gridworld.sanity_checks ({e})", False)

    print("[2] neural library sanity")
    try:
        N.sanity(seed=0); check("neural.sanity ran", True)
    except AssertionError as e:
        check(f"neural.sanity ({e})", False)

    print("[3] trajectory accounting")
    check("exact cost = m_in+m_h+m_out", per_step_cost('exact', 10, 7, 5) == 22)
    check("fo cost = m_in+m_out (no curvature batch)", per_step_cost('fo', 10, 7, 5) == 15)
    check("fedavg cost = m_in+m_h+m_out", per_step_cost('fedavg', 10, 7, 5) == 22)

    print("[4] determinism (same seed -> identical)")
    mdps = G.make_agents()
    r1 = G.train(mdps, 'exact', K=8, tau=3, alpha=2.0, beta=0.3,
                 m_in=10, m_h=10, m_out=10, r=len(mdps), seed=1, eval_every=8)
    r2 = G.train(mdps, 'exact', K=8, tau=3, alpha=2.0, beta=0.3,
                 m_in=10, m_h=10, m_out=10, r=len(mdps), seed=1, eval_every=8)
    check("tabular train reproducible", np.allclose(r1['Fpost'], r2['Fpost']))

    tg, _, _ = make_arc_family()
    t1 = N.train(tg, 'fo', K=6, tau=3, alpha=1.0, beta=0.2, m_in=10, m_h=10,
                 m_out=10, r=len(tg), seed=1, eval_every=6)['theta']
    t2 = N.train(tg, 'fo', K=6, tau=3, alpha=1.0, beta=0.2, m_in=10, m_h=10,
                 m_out=10, r=len(tg), seed=1, eval_every=6)['theta']
    check("neural train reproducible", np.allclose(t1, t2))

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
