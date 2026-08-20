from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR, SVR

from src.config import (
    EXPERIMENT_LOG_PATH,
    ID_COL,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.models_v5 import make_pipeline
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v51_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FEATURE_SET = "S2_core_derived"
POSTPROCESSORS = {
    "raw": lambda x: np.asarray(x, dtype=float),
    "clip_0_1": clip_0_1,
    "clip_0_1_round2": clip_round_2,
}


def kernel_specs():
    specs = [
        (
            "linear",
            "linearsvr_raw",
            LinearSVR(C=1.0, epsilon=0.0, random_state=RANDOM_STATE, max_iter=20000),
            "raw",
            "LinearSVR raw target; additive scorecard stress test",
        ),
    ]
    for c in [0.5, 1.0, 3.0]:
        specs.append(
            (
                "poly_degree2",
                f"svr_poly2_C{str(c).replace('.', 'p')}_scale",
                SVR(kernel="poly", degree=2, C=c, gamma="scale", epsilon=0.0, cache_size=500),
                "raw",
                "Polynomial degree 2 raw target; explicit low-order interaction test",
            )
        )
    specs.append(
        (
            "poly_degree3",
            "svr_poly3_C1_scale",
            SVR(kernel="poly", degree=3, C=1.0, gamma="scale", epsilon=0.0, cache_size=500),
            "raw",
            "Polynomial degree 3 raw target; lightweight check",
        )
    )
    specs.extend(
        [
            (
                "rbf",
                "svr_rbf_raw",
                SVR(
                    kernel="rbf",
                    C=3.963530707518144,
                    gamma=1.0631617004546035,
                    epsilon=0.0,
                    shrinking=True,
                    cache_size=500,
                ),
                "raw",
                "RBF SVR raw target; preferred report-first submission",
            ),
            (
                "rbf",
                "svr_rbf_quantile",
                SVR(
                    kernel="rbf",
                    C=3.963530707518144,
                    gamma=1.0631617004546035,
                    epsilon=0.0,
                    shrinking=True,
                    cache_size=500,
                ),
                "quantile_target",
                "RBF SVR quantile target; target scale refinement check",
            ),
        ]
    )
    return specs


def evaluate_kernel(train_df, spec):
    kernel_type, model_name, estimator, target_mode, notes = spec
    pipeline = make_pipeline(FEATURE_SET, estimator, target_mode, scale=True, dense=True)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    splitter = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold

    rows = []
    oof_parts = []
    for postprocess, post_func in POSTPROCESSORS.items():
        pred = post_func(raw_oof)
        fold_maes = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fold_maes.append(mean_absolute_error(y[mask], pred[mask]))
        rows.append(
            {
                "exp_id": EXP_ID,
                "kernel_type": kernel_type,
                "model": model_name,
                "target_mode": target_mode,
                "postprocess": postprocess,
                "mean_mae": float(np.mean(fold_maes)),
                "std_mae": float(np.std(fold_maes, ddof=1)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "notes": notes,
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    ID_COL: train_df[ID_COL],
                    TARGET: train_df[TARGET],
                    "exp_id": EXP_ID,
                    "kernel_type": kernel_type,
                    "model": model_name,
                    "target_mode": target_mode,
                    "postprocess": postprocess,
                    "fold": folds,
                    "oof_pred": pred,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(oof_parts, ignore_index=True)


def append_experiment_log(comparison):
    log_df = comparison.rename(
        columns={
            "kernel_type": "feature_version",
            "pred_std": "pred_std_oof",
            "pred_min": "pred_min_oof",
            "pred_max": "pred_max_oof",
        }
    )
    log_df["notes"] = (
        log_df["notes"]
        + "; feature_set="
        + FEATURE_SET
        + "; target_mode="
        + log_df["target_mode"]
    )
    keep_cols = [
        "exp_id",
        "feature_version",
        "model",
        "postprocess",
        "mean_mae",
        "std_mae",
        "pred_std_oof",
        "pred_min_oof",
        "pred_max_oof",
        "notes",
    ]
    log_df = log_df[keep_cols].copy()
    if EXPERIMENT_LOG_PATH.exists():
        existing = pd.read_csv(EXPERIMENT_LOG_PATH)
        for col in log_df.columns:
            if col not in existing.columns:
                existing[col] = np.nan
        for col in existing.columns:
            if col not in log_df.columns:
                log_df[col] = np.nan
        combined = pd.concat([existing, log_df[existing.columns]], ignore_index=True)
    else:
        combined = log_df
    combined.to_csv(EXPERIMENT_LOG_PATH, index=False)


def make_submission(train_df, test_df, sample_submission, estimator, target_mode, postprocess, path):
    pipeline = make_pipeline(FEATURE_SET, estimator, target_mode, scale=True, dense=True)
    pipeline.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    pred = POSTPROCESSORS[postprocess](pipeline.predict(test_df))
    submission = sample_submission.copy()
    submission[TARGET] = pred
    submission.to_csv(path, index=False)


def run_v51_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    all_results = []
    all_oof = []
    specs = kernel_specs()
    for spec in specs:
        print(f"[{spec[1]}] target={spec[3]}")
        result_df, oof_df = evaluate_kernel(train_df, spec)
        print(
            result_df[["postprocess", "mean_mae", "std_mae", "pred_std"]]
            .round(6)
            .to_string(index=False)
        )
        all_results.append(result_df)
        all_oof.append(oof_df)

    comparison = pd.concat(all_results, ignore_index=True).sort_values("mean_mae").reset_index(drop=True)
    oof_output = pd.concat(all_oof, ignore_index=True)

    reports_dir = EXPERIMENT_LOG_PATH.parent
    comparison_path = reports_dir / "v51_kernel_comparison.csv"
    oof_path = reports_dir / "oof_predictions_v51.csv"
    comparison.to_csv(comparison_path, index=False)
    oof_output.to_csv(oof_path, index=False)
    append_experiment_log(comparison)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    raw_spec = next(s for s in specs if s[1] == "svr_rbf_raw")
    quantile_spec = next(s for s in specs if s[1] == "svr_rbf_quantile")
    raw_path = SUBMISSIONS_DIR / "v51_svr_rbf_S2_core_derived_raw_target_clip_0_1_round2.csv"
    quantile_path = SUBMISSIONS_DIR / "v51_svr_rbf_S2_core_derived_quantile_target_clip_0_1_round2.csv"
    make_submission(train_df, test_df, sample_submission, raw_spec[2], "raw", "clip_0_1_round2", raw_path)
    make_submission(
        train_df,
        test_df,
        sample_submission,
        quantile_spec[2],
        "quantile_target",
        "clip_0_1_round2",
        quantile_path,
    )

    print("\n=== V5.1 kernel comparison ===")
    print(comparison.round(6).to_string(index=False))
    linear_best = comparison[comparison["kernel_type"].eq("linear")].iloc[0]
    poly_best = comparison[comparison["kernel_type"].str.startswith("poly")].iloc[0]
    rbf_raw_best = comparison[comparison["model"].eq("svr_rbf_raw")].iloc[0]
    rbf_quantile_best = comparison[comparison["model"].eq("svr_rbf_quantile")].iloc[0]
    print("\n=== Interpretation ===")
    print(f"LinearSVR best MAE: {linear_best['mean_mae']:.6f}")
    print(f"Polynomial SVR best MAE: {poly_best['mean_mae']:.6f}")
    print(f"RBF raw target best MAE: {rbf_raw_best['mean_mae']:.6f}")
    print(f"RBF quantile target best MAE: {rbf_quantile_best['mean_mae']:.6f}")
    print("Raw target RBF submission is the report-first candidate; quantile target is a scale-refinement check.")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved OOF: {oof_path}")
    print(f"Saved raw target submission: {raw_path}")
    print(f"Saved quantile target submission: {quantile_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")
    return comparison, oof_output, raw_path


if __name__ == "__main__":
    run_v51_experiments()
