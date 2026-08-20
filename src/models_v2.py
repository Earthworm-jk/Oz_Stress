from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    EXPERIMENT_LOG_PATH,
    ID_COL,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    V2_CATEGORICAL_COLS,
)
from src.features import (
    CategoricalStringCaster,
    InteractionFeatureEngineer,
    QuantileScorecardBinner,
    StressFeatureEngineer,
    make_ohe_preprocessor,
)
from src.postprocess import POSTPROCESSORS
from src.validation import evaluate_pipeline_with_oof


FEATURE_VERSION = "v2_scorecard_interactions"
EXP_ID = f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class CatBoostNativeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_cols=None, params=None):
        self.categorical_cols = categorical_cols
        self.params = params

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        X_df = X.copy()
        categorical_cols = self.categorical_cols or V2_CATEGORICAL_COLS
        self.cat_features_ = [col for col in categorical_cols if col in X_df.columns]
        params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "iterations": 1200,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
        }
        if self.params:
            params.update(self.params)
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(X_df, y, cat_features=self.cat_features_)
        return self

    def predict(self, X):
        return self.model_.predict(X.copy())


def make_v2_feature_steps():
    return [
        ("features", StressFeatureEngineer()),
        ("scorecard_bins", QuantileScorecardBinner(n_bins=5)),
        ("interactions", InteractionFeatureEngineer()),
    ]


def make_v2_ohe_pipeline(estimator, scale_numeric=False):
    return Pipeline(
        steps=[
            *make_v2_feature_steps(),
            (
                "preprocess",
                make_ohe_preprocessor(V2_CATEGORICAL_COLS, scale_numeric=scale_numeric),
            ),
            ("model", estimator),
        ]
    )


def make_catboost_native_pipeline():
    return Pipeline(
        steps=[
            *make_v2_feature_steps(),
            ("categorical_strings", CategoricalStringCaster(V2_CATEGORICAL_COLS)),
            ("model", CatBoostNativeRegressor(V2_CATEGORICAL_COLS)),
        ]
    )


def get_v2_model_specs():
    specs = [
        (
            "ridge_ohe_v2",
            make_v2_ohe_pipeline(Ridge(alpha=10.0, random_state=RANDOM_STATE), scale_numeric=True),
            "Ridge with OHE plus scorecard bins and interactions",
        ),
        (
            "extratrees_ohe_v2",
            make_v2_ohe_pipeline(
                ExtraTreesRegressor(
                    n_estimators=700,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "ExtraTrees with OHE plus scorecard bins and interactions",
        ),
    ]

    try:
        import catboost  # noqa: F401

        specs.append(
            (
                "catboost_native_v2",
                make_catboost_native_pipeline(),
                "Native CatBoost categorical handling; no OneHotEncoder",
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        specs.append(
            (
                "lightgbm_ohe_v2",
                make_v2_ohe_pipeline(
                    LGBMRegressor(
                        objective="regression_l1",
                        n_estimators=900,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    )
                ),
                "LightGBM with OHE plus scorecard bins and interactions",
            )
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        specs.append(
            (
                "xgboost_ohe_v2",
                make_v2_ohe_pipeline(
                    XGBRegressor(
                        objective="reg:absoluteerror",
                        n_estimators=800,
                        learning_rate=0.03,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                        eval_metric="mae",
                    )
                ),
                "XGBoost with OHE plus scorecard bins and interactions",
            )
        )
    except ImportError:
        pass

    return specs


def prediction_distribution(label, values):
    arr = np.asarray(values, dtype=float)
    return {
        f"{label}_mean": float(np.mean(arr)),
        f"{label}_std": float(np.std(arr, ddof=1)),
        f"{label}_min": float(np.min(arr)),
        f"{label}_max": float(np.max(arr)),
    }


def print_distribution_diagnostics(train_df, oof_df, submission_pred, best_model, best_postprocess):
    target_stats = prediction_distribution("train_target", train_df[TARGET])
    best_oof = oof_df[
        (oof_df["model"] == best_model) & (oof_df["postprocess"] == best_postprocess)
    ]["oof_pred"]
    oof_stats = prediction_distribution("oof_pred", best_oof)
    sub_stats = prediction_distribution("submission_pred", submission_pred)

    print("\n=== Prediction distribution diagnostics ===")
    for stats in (target_stats, oof_stats, sub_stats):
        print(pd.Series(stats).round(6).to_string())

    target_std = target_stats["train_target_std"]
    oof_std = oof_stats["oof_pred_std"]
    ratio = oof_std / target_std if target_std else np.nan
    print(f"OOF std / target std: {ratio:.3f}")
    if ratio < 0.50:
        print("Warning: OOF predictions look strongly shrunk toward the mean.")
    elif ratio < 0.70:
        print("Notice: OOF predictions are moderately shrunk toward the mean.")
    else:
        print("OOF prediction spread does not look severely mean-shrunk.")


def append_v2_experiment_log(results_df):
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def run_v2_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    all_fold_results = []
    all_summary_results = []
    all_oof_results = []
    model_specs = get_v2_model_specs()
    notes_by_model = {model_name: notes for model_name, _, notes in model_specs}

    for model_name, pipeline, _ in model_specs:
        print(f"\n[{model_name}]")
        fold_df, summary_df, oof_df = evaluate_pipeline_with_oof(model_name, pipeline, train_df)
        print(fold_df.pivot(index="fold", columns="postprocess", values="mae").round(6))
        print(summary_df.round(6))
        all_fold_results.append(fold_df)
        all_summary_results.append(summary_df)
        all_oof_results.append(oof_df)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    oof_results = pd.concat(all_oof_results, ignore_index=True)
    results_df = (
        pd.concat(all_summary_results, ignore_index=True)
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    results_df.insert(0, "feature_version", FEATURE_VERSION)
    results_df.insert(0, "exp_id", EXP_ID)
    results_df["notes"] = results_df["model"].map(notes_by_model)

    reports_dir = EXPERIMENT_LOG_PATH.parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    oof_path = reports_dir / "oof_predictions_v2.csv"
    oof_results[[ID_COL, TARGET, "model", "postprocess", "oof_pred", "fold"]].to_csv(
        oof_path, index=False
    )

    best = results_df.iloc[0]
    best_name = best["model"]
    best_postprocess = best["postprocess"]
    best_pipeline = clone({model_name: pipe for model_name, pipe, _ in model_specs}[best_name])

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    best_pipeline.fit(X_train, y_train)
    submission_pred = POSTPROCESSORS[best_postprocess](best_pipeline.predict(test_df))

    submission = sample_submission.copy()
    submission[TARGET] = submission_pred
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / f"{EXP_ID}_{best_name}_{best_postprocess}.csv"
    submission.to_csv(submission_path, index=False)

    append_v2_experiment_log(results_df)
    print_distribution_diagnostics(
        train_df, oof_results, submission_pred, best_name, best_postprocess
    )

    print("\n=== V2 model comparison ===")
    print(results_df.round(6))
    print(f"\nBest: {best_name} / {best_postprocess} / CV MAE={best['mean_mae']:.6f}")
    print(f"Saved OOF: {oof_path}")
    print(f"Saved submission: {submission_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")

    return fold_results, results_df, oof_results, submission_path


if __name__ == "__main__":
    run_v2_experiments()
