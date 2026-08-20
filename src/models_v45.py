from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.config import (
    EXPERIMENT_LOG_PATH,
    ID_COL,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    V3_CATEGORICAL_COLS,
)
from src.features import (
    InteractionFeatureEngineer,
    QuantileScorecardBinner,
    StressFeatureEngineer,
    make_ohe_preprocessor,
)
from src.postprocess import clip_0_1, clip_round_2
from src.validation import make_cv


FEATURE_VERSION = "v45_v3_exact_multiseed"
EXP_ID = f"v45_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SEEDS = [42, 2024, 777, 1004, 2026]
V3_SINGLE_BEST_MAE = 0.179860
V3_OOF_STD = 0.175966


def make_v3_exact_pipeline(seed):
    return Pipeline(
        steps=[
            ("features", StressFeatureEngineer(add_mean_working_cat_v3=True)),
            ("scorecard_bins", QuantileScorecardBinner(n_bins=5)),
            ("interactions", InteractionFeatureEngineer()),
            ("preprocess", make_ohe_preprocessor(V3_CATEGORICAL_COLS)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=1000,
                    min_samples_leaf=1,
                    max_features=0.8,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def summarize_prediction(model_name, postprocess, pred, y, folds):
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_rows.append(
            {
                "model": model_name,
                "postprocess": postprocess,
                "fold": int(fold),
                "mae": mean_absolute_error(y[mask], pred[mask]),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "model": model_name,
        "postprocess": postprocess,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std()),
        "pred_std_oof": float(np.std(pred, ddof=1)),
        "pred_min_oof": float(np.min(pred)),
        "pred_max_oof": float(np.max(pred)),
    }
    return summary, fold_df


def append_experiment_log(results_df):
    log_df = results_df[
        [
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
    ].copy()
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


def run_v45_experiment():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)

    model_name = "extratrees_v45_v3_exact_5seed"
    for fold, (tr_idx, va_idx) in enumerate(make_cv(y), start=1):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train = y[tr_idx]
        seed_preds = []
        for seed in SEEDS:
            model = make_v3_exact_pipeline(seed)
            model.fit(X_train, y_train)
            seed_preds.append(model.predict(X_valid))
        raw_oof[va_idx] = np.mean(seed_preds, axis=0)
        folds[va_idx] = fold
        print(f"fold {fold}: done")

    oof_frames = []
    summaries = []
    fold_frames = []
    postprocessed = {
        "clip_0_1": clip_0_1(raw_oof),
        "clip_0_1_round2": clip_round_2(raw_oof),
    }
    for postprocess, pred in postprocessed.items():
        summary, fold_df = summarize_prediction(model_name, postprocess, pred, y, folds)
        summaries.append(summary)
        fold_frames.append(fold_df)
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COL: train_df[ID_COL],
                    TARGET: train_df[TARGET],
                    "model": model_name,
                    "postprocess": postprocess,
                    "oof_pred": pred,
                    "fold": folds,
                }
            )
        )

    comparison = pd.DataFrame(summaries).sort_values("mean_mae").reset_index(drop=True)
    comparison.insert(0, "feature_version", FEATURE_VERSION)
    comparison.insert(0, "exp_id", EXP_ID)
    comparison["notes"] = (
        f"V3 exact feature pipeline; ExtraTrees seeds averaged {SEEDS}; no target encoding"
    )

    oof_output = pd.concat(oof_frames, ignore_index=True)
    reports_dir = EXPERIMENT_LOG_PATH.parent
    oof_path = reports_dir / "oof_predictions_v45.csv"
    comparison_path = reports_dir / "v45_model_comparison.csv"
    oof_output[[ID_COL, TARGET, "model", "postprocess", "oof_pred", "fold"]].to_csv(
        oof_path, index=False
    )
    comparison.to_csv(comparison_path, index=False)
    append_experiment_log(comparison)

    best = comparison.iloc[0]
    raw_test_preds = []
    for seed in SEEDS:
        model = make_v3_exact_pipeline(seed)
        model.fit(X, y)
        raw_test_preds.append(model.predict(test_df))
    raw_test_pred = np.mean(raw_test_preds, axis=0)
    submission_pred = (
        clip_round_2(raw_test_pred)
        if best["postprocess"] == "clip_0_1_round2"
        else clip_0_1(raw_test_pred)
    )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = submission_pred
    submission_path = SUBMISSIONS_DIR / f"{EXP_ID}_best_{model_name}_{best['postprocess']}.csv"
    submission.to_csv(submission_path, index=False)

    std_ratio = best["pred_std_oof"] / V3_OOF_STD
    decision = (
        "v45 is a submission candidate."
        if best["mean_mae"] < V3_SINGLE_BEST_MAE
        else "Keep existing v3 round2 as the submission candidate."
    )

    print("\n=== V45 comparison ===")
    print(comparison.round(6).to_string(index=False))
    print(f"\nV3 single best MAE: {V3_SINGLE_BEST_MAE:.6f}")
    print(f"V45 best MAE: {best['mean_mae']:.6f}")
    print(f"Delta vs v3: {best['mean_mae'] - V3_SINGLE_BEST_MAE:+.6f}")
    print(f"V3 OOF std reference: {V3_OOF_STD:.6f}")
    print(f"V45 best OOF std: {best['pred_std_oof']:.6f}")
    print(f"OOF std ratio vs v3: {std_ratio:.3f}")
    if std_ratio < 0.90:
        print("Warning: ensemble predictions are materially more mean-shrunk than v3.")
    else:
        print("OOF std is not excessively reduced versus v3.")
    print(decision)
    print(f"Saved OOF: {oof_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved submission: {submission_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")

    return comparison, oof_output, submission_path


if __name__ == "__main__":
    run_v45_experiment()
