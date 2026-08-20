from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
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
    V4_CATEGORICAL_COLS,
    V4_INTERACTION_SPECS,
    V4_TARGET_ENCODING_COLS,
)
from src.features import (
    InteractionFeatureEngineer,
    QuantileScorecardBinner,
    StressFeatureEngineer,
    make_ohe_preprocessor,
)
from src.postprocess import clip_0_1, clip_round_2
from src.validation import evaluate_pipeline_with_oof, make_cv


FEATURE_VERSION = "v4_leaf1_target_encoding_ensemble"
EXP_ID = f"v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SEEDS = [42, 2024, 777, 1004, 2026]


class SmoothedTargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, smoothing=20.0, missing_token="__MISSING__"):
        self.columns = columns
        self.smoothing = smoothing
        self.missing_token = missing_token

    def fit(self, X, y):
        X_df = self._to_frame(X)
        y_s = pd.Series(np.asarray(y, dtype=float), index=X_df.index)
        self.global_mean_ = float(y_s.mean())
        self.mappings_ = {}
        for col in self.columns:
            keys = self._keys(X_df[col])
            stats = y_s.groupby(keys).agg(["mean", "count"])
            smooth = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.mappings_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X_df = self._to_frame(X).copy()
        for col in self.columns:
            encoded = self._keys(X_df[col]).map(self.mappings_[col]).fillna(self.global_mean_)
            X_df[f"{col}__te"] = encoded.astype(float)
        return X_df

    def _keys(self, values):
        return values.astype("object").fillna(self.missing_token).astype(str)

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def make_v4_pipeline(max_features, n_estimators=1000, seed=RANDOM_STATE, use_te=True):
    steps = [
        ("features", StressFeatureEngineer(add_mean_working_cat_v3=True)),
        ("scorecard_bins", QuantileScorecardBinner(n_bins=5)),
        ("interactions", InteractionFeatureEngineer()),
        ("v4_interactions", InteractionFeatureEngineer(V4_INTERACTION_SPECS)),
    ]
    if use_te:
        steps.append(
            (
                "target_encoding",
                SmoothedTargetMeanEncoder(V4_TARGET_ENCODING_COLS, smoothing=20.0),
            )
        )
    steps.extend(
        [
            ("preprocess", make_ohe_preprocessor(V4_CATEGORICAL_COLS)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=1,
                    max_features=max_features,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return Pipeline(steps=steps)


def tuning_specs():
    specs = []
    for n_estimators in [1000]:
        for max_features in [0.6, 0.7, 0.8, 0.9, 1.0]:
            name = f"extratrees_v4_mf{str(max_features).replace('.', '')}_n{n_estimators}"
            specs.append((name, max_features, n_estimators))
    return specs


def summarize_oof(model_name, postprocess, pred, y, fold_numbers):
    rows = []
    for fold in sorted(np.unique(fold_numbers)):
        mask = fold_numbers == fold
        rows.append(
            {
                "model": model_name,
                "postprocess": postprocess,
                "fold": int(fold),
                "mae": mean_absolute_error(y[mask], pred[mask]),
            }
        )
    fold_df = pd.DataFrame(rows)
    return {
        "model": model_name,
        "postprocess": postprocess,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std()),
        "pred_std_oof": float(np.std(pred, ddof=1)),
        "pred_min_oof": float(np.min(pred)),
        "pred_max_oof": float(np.max(pred)),
    }, fold_df


def evaluate_seed_ensemble(model_name, max_features, n_estimators, train_df):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    fold_numbers = np.zeros(len(train_df), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(make_cv(y), start=1):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train = y[tr_idx]
        seed_preds = []
        for seed in SEEDS:
            model = make_v4_pipeline(max_features, n_estimators, seed=seed, use_te=True)
            model.fit(X_train, y_train)
            seed_preds.append(model.predict(X_valid))
        raw_oof[va_idx] = np.mean(seed_preds, axis=0)
        fold_numbers[va_idx] = fold

    oof_frames = []
    summaries = []
    fold_frames = []
    for postprocess, pred in [
        ("clip_0_1", clip_0_1(raw_oof)),
        ("clip_0_1_round2", clip_round_2(raw_oof)),
    ]:
        summary, fold_df = summarize_oof(model_name, postprocess, pred, y, fold_numbers)
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
                    "fold": fold_numbers,
                }
            )
        )

    return pd.concat(fold_frames, ignore_index=True), pd.DataFrame(summaries), pd.concat(
        oof_frames, ignore_index=True
    )


def fit_predict_seed_ensemble(max_features, n_estimators, train_df, test_df):
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()
    preds = []
    for seed in SEEDS:
        model = make_v4_pipeline(max_features, n_estimators, seed=seed, use_te=True)
        model.fit(X_train, y_train)
        preds.append(model.predict(test_df))
    return np.mean(preds, axis=0)


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


def run_v4_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    all_fold = []
    all_summary = []
    all_oof = []
    spec_lookup = {}

    for model_name, max_features, n_estimators in tuning_specs():
        print(f"\n[{model_name}]")
        pipeline = make_v4_pipeline(max_features, n_estimators, seed=RANDOM_STATE, use_te=True)
        fold_df, summary_df, oof_df = evaluate_pipeline_with_oof(model_name, pipeline, train_df)
        summary_df = summary_df[summary_df["postprocess"].isin(["clip_0_1", "clip_0_1_round2"])]
        print(fold_df.pivot(index="fold", columns="postprocess", values="mae").round(6))
        print(summary_df.round(6))
        all_fold.append(fold_df)
        all_summary.append(summary_df)
        all_oof.append(oof_df[oof_df["postprocess"].isin(["clip_0_1", "clip_0_1_round2"])])
        spec_lookup[model_name] = (max_features, n_estimators)

    tuning_results = pd.concat(all_summary, ignore_index=True).sort_values("mean_mae").reset_index(drop=True)
    best_single = tuning_results.iloc[0]
    best_model_name = best_single["model"]
    best_max_features, best_n_estimators = spec_lookup[best_model_name]

    ensemble_name = f"extratrees_v4_ensemble_{best_model_name}"
    print(f"\n[{ensemble_name}] seeds={SEEDS}")
    ens_fold, ens_summary, ens_oof = evaluate_seed_ensemble(
        ensemble_name, best_max_features, best_n_estimators, train_df
    )
    print(ens_fold.pivot(index="fold", columns="postprocess", values="mae").round(6))
    print(ens_summary.round(6))

    comparison = pd.concat([tuning_results, ens_summary], ignore_index=True)
    comparison.insert(0, "feature_version", FEATURE_VERSION)
    comparison.insert(0, "exp_id", EXP_ID)
    comparison["notes"] = np.where(
        comparison["model"].eq(ensemble_name),
        f"Multi-seed ensemble {SEEDS}; target encoding",
        "ExtraTrees leaf1 max_features tuning at n_estimators=1000; target encoding",
    )
    comparison = comparison.sort_values("mean_mae").reset_index(drop=True)

    oof_output = pd.concat([*all_oof, ens_oof], ignore_index=True)
    reports_dir = EXPERIMENT_LOG_PATH.parent
    oof_path = reports_dir / "oof_predictions_v4.csv"
    comparison_path = reports_dir / "v4_model_comparison.csv"
    oof_output[[ID_COL, TARGET, "model", "postprocess", "oof_pred", "fold"]].to_csv(oof_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    append_experiment_log(comparison)

    best = comparison.iloc[0]
    best_name = best["model"]
    best_postprocess = best["postprocess"]
    if best_name == ensemble_name:
        raw_test_pred = fit_predict_seed_ensemble(best_max_features, best_n_estimators, train_df, test_df)
    else:
        max_features, n_estimators = spec_lookup[best_name]
        model = make_v4_pipeline(max_features, n_estimators, seed=RANDOM_STATE, use_te=True)
        model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
        raw_test_pred = model.predict(test_df)

    pred = clip_0_1(raw_test_pred)
    if best_postprocess == "clip_0_1_round2":
        pred = clip_round_2(raw_test_pred)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = pred
    submission_path = SUBMISSIONS_DIR / f"{EXP_ID}_best_{best_name}_{best_postprocess}.csv"
    submission.to_csv(submission_path, index=False)

    second_path = None
    if best_name == ensemble_name:
        alt_postprocess = "clip_0_1_round2" if best_postprocess == "clip_0_1" else "clip_0_1"
        alt_pred = clip_round_2(raw_test_pred) if alt_postprocess == "clip_0_1_round2" else clip_0_1(raw_test_pred)
        alt_submission = sample_submission.copy()
        alt_submission[TARGET] = alt_pred
        second_path = SUBMISSIONS_DIR / f"{EXP_ID}_best_{best_name}_{alt_postprocess}.csv"
        alt_submission.to_csv(second_path, index=False)

    v3_best_mae = 0.179860
    print("\n=== V4 model comparison ===")
    print(
        comparison[
            [
                "model",
                "postprocess",
                "mean_mae",
                "std_mae",
                "pred_std_oof",
                "pred_min_oof",
                "pred_max_oof",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nV3 best MAE: {v3_best_mae:.6f}")
    print(f"V4 best MAE: {best['mean_mae']:.6f}")
    print(f"Delta vs v3: {best['mean_mae'] - v3_best_mae:+.6f}")
    print(f"Saved OOF: {oof_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved submission: {submission_path}")
    if second_path:
        print(f"Saved second submission: {second_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")

    return comparison, oof_output, submission_path


if __name__ == "__main__":
    run_v4_experiments()
