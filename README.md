A small JAX kernel of DISCO (arXiv 2609.39326) for causal discovery from count data.

Install and run from this directory.

pip install -e .
PYTHONPATH=src python -m pytest -q
python train.py

Oracle conditional and off-diagonal curvature scores recover a 3-node chain. The parent threshold is tau = 0.05. train.py prints loss_start and loss_end for 300 Adam steps and exits 0 only if loss_end is under 90 percent of loss_start. It does not print a causal order from the fitted network.

What does not match the paper: no 1000-node DAG recovery, no Poisson, negative binomial, or binomial benchmark tables, and no four-level quantile representation at paper scale. The score network is a tiny MLP rather than the paper's discrete diffusion architecture. Marginal concrete-score projection (Proposition 10) is not a separate network. Oracle tests use the exact marginal.
