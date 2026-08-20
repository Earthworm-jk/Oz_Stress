from datetime import datetime

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

from src.config import (
    EXPERIMENT_LOG_PATH,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.features import StressFeatureEngineer, make_preprocessor
from src.postprocess import POSTPROCESSORS
from src.validation import evaluate_pipeline


def make_pipeline(estimator, scale_numeric=False):
    return Pipeline(
        steps=[
            ("features", StressFeatureEngineer()),
            ("preprocess", make_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def get_model_specs():
    specs = [
        (
            "ridge_ohe",
            make_pipeline(Ridge(alpha=10.0, random_state=RANDOM_STATE), scale_numeric=True),
        ),
        (
            "elasticnet_ohe",
            make_pipeline(
                ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000, random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
        ),
        (
            "extratrees_ohe",
            make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=600,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ),
    ]

    try:
        from catboost import CatBoostRegressor

        specs.append(
            (
                "catboost_ohe",
                make_pipeline(
                    CatBoostRegressor(
                        loss_function="MAE",
                        eval_metric="MAE",
                        iterations=800,
                        learning_rate=0.03,
                        depth=6,
                        random_seed=RANDOM_STATE,
                        verbose=False,
                        allow_writing_files=False,
                    )
                ),
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        specs.append(
            (
                "lightgbm_ohe",
                make_pipeline(
                    LGBMRegressor(
                        objective="regression_l1",
                        n_estimators=800,
                        learning_rate=0.03,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    )
                ),
            )
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        specs.append(
            (
                "xgboost_ohe",
                make_pipeline(
                    XGBRegressor(
                        objective="reg:absoluteerror",
                        n_estimators=700,
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
            )
        )
    except ImportError:
        pass

    return specs


def append_experiment_log(results_df, submission_path):
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df = results_df.copy()
    log_df.insert(0, "run_at", datetime.now().isoformat(timespec="seconds"))
    log_df["submission_path"] = str(submission_path)
    write_header = not EXPERIMENT_LOG_PATH.exists()
    log_df.to_csv(EXPERIMENT_LOG_PATH, mode="a", header=write_header, index=False)


def run_baseline():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    all_fold_results = []
    all_summary_results = []
    model_specs = get_model_specs()

    for model_name, pipeline in model_specs:
        print(f"\n[{model_name}]")
        fold_df, summary_df = evaluate_pipeline(model_name, pipeline, train_df)
        print(fold_df.pivot(index="fold", columns="postprocess", values="mae").round(6))
        print(summary_df.round(6))
        all_fold_results.append(fold_df)
        all_summary_results.append(summary_df)

    fold_results = pd.concat(all_fold_results, ignore_index=True)
    results_df = pd.concat(all_summary_results, ignore_index=True).sort_values("mean_mae").reset_index(drop=True)

    best = results_df.iloc[0]
    best_name = best["model"]
    best_postprocess = best["postprocess"]
    best_pipeline = dict(model_specs)[best_name]

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    best_pipeline.fit(X_train, y_train)
    test_pred = best_pipeline.predict(test_df)
    test_pred = POSTPROCESSORS[best_postprocess](test_pred)

    submission = sample_submission.copy()
    submission[TARGET] = test_pred

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = SUBMISSIONS_DIR / f"{best_name}_{best_postprocess}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    append_experiment_log(results_df, submission_path)

    print("\n=== Model comparison ===")
    print(results_df.round(6))
    print(f"\nBest: {best_name} / {best_postprocess} / CV MAE={best['mean_mae']:.6f}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved experiment log: {EXPERIMENT_LOG_PATH}")

    return fold_results, results_df, submission_path


if __name__ == "__main__":
    run_baseline()
