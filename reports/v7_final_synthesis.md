# v7 Neural Baseline Synthesis

## Result
Best MLPRegressor: `mlp_wide_relu_l2_1e3` with `clip_0_1_round2` / CV MAE `0.218363`.

Current raw RBF sentinel99 reference CV MAE: `0.134917`.

## Interpretation
1. MLPRegressor did not approach the RBF SVR sentinel branch if its CV remains materially above the reference.
2. Neural models are nonlinear and improve over weak linear/additive baselines only if their CV is below the linear range, but the stable RBF kernel remains stronger for this 3,000-row tabular setting.
3. The prediction distribution and OOF scatter figures are saved in `figures/`.
4. PyTorch decision: PyTorch skipped: torch is unavailable (ModuleNotFoundError).
5. Submission created: `False`. The v7 branch is primarily report evidence, not a submission branch.

## Model family comparison

```csv
family,model,cv_mae,notes
kernel,raw RBF sentinel99,0.13491666666666666,v5.3/v5.4 sentinel branch.
kernel,raw RBF S2,0.139413,v5.1 raw RBF S2.
tree,ExtraTrees v3,0.17986,Best tree-style scorecard branch.
neural,mlp_wide_relu_l2_1e3,0.21836333333333333,Best v7 MLPRegressor.
linear,Ridge/ElasticNet/LinearSVR,0.2445,Earlier baseline range; weak additive scorecard.

```
