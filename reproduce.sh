#!/usr/bin/env bash
# Reproduce all experiment data and figures. Run from the repo root.
set -e
cd "$(dirname "$0")"
echo "[tests]";     python3 run_tests.py
echo "[E1/E2]";     python3 scripts/e1e2_tabular.py
echo "[E3]";        for v in exact fo fedavg; do python3 scripts/e3_alpha.py "$v"; done
                    python3 scripts/e3_alpha.py merge
echo "[E4]";        for mh in 10 50 100 200 500 1000; do python3 scripts/e4_curvature_neural.py "$mh"; done
                    python3 scripts/e4_curvature_neural.py fedavg
                    python3 scripts/e4_curvature_neural.py merge
echo "[E5]";        python3 scripts/e5_fewshot_neural.py
echo "[E7]";        python3 scripts/e7_trajectories_neural.py
echo "[figures]";   python3 scripts/make_figures.py
echo "Done. Figures in figures/, data in results/."
