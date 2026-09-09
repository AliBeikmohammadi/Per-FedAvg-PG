.PHONY: install test figures tabular neural all clean

install:
	pip install -r requirements.txt

test:
	python run_tests.py

figures:
	python scripts/make_figures.py

tabular:
	python scripts/e1e2_tabular.py
	for v in exact fo fedavg; do python scripts/e3_alpha.py $$v; done
	python scripts/e3_alpha.py merge

neural:
	for mh in 10 50 100 200 500 1000; do python scripts/e4_curvature_neural.py $$mh; done
	python scripts/e4_curvature_neural.py fedavg
	python scripts/e4_curvature_neural.py merge
	python scripts/e5_fewshot_neural.py
	python scripts/e7_trajectories_neural.py

all: test tabular neural figures

# --- package:begin -- dev-only; package.py strips this block, and this
# comment, out of the Makefile it copies into either release zip, since
# package.py itself (which these targets call) never ships in either one. ---
.PHONY: package package-github package-blind

package: package-github package-blind

package-github:
	python package.py github

package-blind:
	python package.py blind
# --- package:end ---

clean:
	rm -rf $$(find . -name __pycache__) figures/*.png

