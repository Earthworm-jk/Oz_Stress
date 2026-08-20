# v6.1 Final Synthesis

## 1. RBF local tuning
Best local-tuning row: `c0.7_g1.15_e0` / `round2` with CV MAE `0.134903`.
Baseline sentinel99 CV MAE is approximately `0.134917`. A submission candidate is created only if improvement is at least 0.0005.

## 2. Target lattice
The target is an exact 0..100 integer lattice after multiplying by 100. This strengthens the bounded synthetic/grid score hypothesis.

## 3. Symbolic approximation
Sparse polynomial approximations are diagnostic only. If pred100 is easier to approximate than y100, it suggests the RBF is smoothing a latent score surface rather than reconstructing a simple explicit formula.

## 4. Counterfactual probing
Counterfactual reports in `v61_counterfactual_probing.csv` show how the fitted raw RBF reacts to mean_working sentinel values, sleep/activity categories, metabolic variables, blood pressure, and bone density.

## 5. Sample geometry
PCA and distance diagnostics are saved in `v61_sample_geometry_summary.csv` and figures. Missing mean_working geometry is evaluated without looking at test distribution.

## 6. RBF distillation
Distillation fidelity is summarized in `v61_rbf_distillation_fidelity.csv`. This indicates how much of the RBF prediction function can be compressed into explainable tree-like structure.

## 7. Residual topology
Residual clusters and large residual profiles are saved in `v61_residual_topology_clusters.csv` and `v61_large_residual_profile.csv`. These are diagnostic and not direct correction rules.

## 8. Submission decision
Submission created: `False`.
If no submission was created, the safer final candidate remains the v5/v54 raw RBF sentinel candidate.
