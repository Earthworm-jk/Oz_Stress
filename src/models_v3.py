from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
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
    V3_CATEGORICAL_COLS,
)
from src.features import (
    InteractionFeatureEngineer,
    QuantileScorecardBinner,
    StressFeatureEngineer,
    make_ohe_preprocessor,
)
from src.postprocess import clip_0_1, clip_round_2
from src.validation import evaluate_pipeline_with_oof


FEATURE_VERSION = "v3_mwcat_calibrated_extratrees"
EXP_ID = f"v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class NoOpCalibrator(BaseEstimator):
    def fit(self, pred, y):
        return self

    def predict(self, pred):
        return np.asarray(pred, dtype=float)


class ClipCalibrator(BaseEstimator):
    def fit(self, pred, y):
        return self

    def predict(self, pred):
        return clip_0_1(pred)


class HuberPredictionCalibrator(BaseEstimator):
    def __init__(self):
        self.model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000)

    def fit(self, pred, y):
        self.model.fit(np.asarray(pred).reshape(-1, 1), y)
        return self

    def predict(self, pred):
        calibrated = self.model.predict(np.asarray(pred).reshape(-1, 1))
        return clip_0_1(calibrated)


class IsotonicPredictionCalibrator(BaseEstimator):
    def __init__(self):
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, pred, y):
        self.model.fit(np.asarray(pred, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, pred):
        return clip_0_1(self.model.predict(np.asarray(pred, dtype=float)))


class QuantileBinMedianCalibrator(BaseEstimator):
    def __init__(self, n_bins=20):
        self.n_bins = n_bins

    def fit(self, pred, y):
        pred_s = pd.Series(np.asarray(pred, dtype=float))
        y_s = pd.Series(np.asarray(y, dtype=float))
        if pred_s.nunique() <= 1:
            self.edges_ = np.array([-np.inf, np.inf], dtype=float)
            self.bin_values_ = np.array([float(y_s.median())], dtype=float)
            return self

        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = pred_s.quantile(quantiles).to_numpy(dtype=float)
        edges = np.unique(edges)
        edges[0] = -np.inf
        edges[-1] = np.inf
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf], dtype=float)

        bins = pd.cut(pred_s, bins=edges, labels=False, include_lowest=True)
        global_median = float(y_s.median())
        medians = y_s.groupby(bins).median()
        self.edges_ = edges
        self.bin_values_ = np.array(
            [float(medians.get(i, global_median)) for i in range(len(edges) - 1)],
            dtype=float,
        )
        self.global_median_ = global_median
        return self

    def predict(self, pred):
        pred_s = pd.Series(np.asarray(pred, dtype=float))
        bins = pd.cut(pred_s, bins=self.edges_, labels=False, include_lowest=True)
        values = np.full(len(pred_s), self.global_median_, dtype=float)
        valid = bins.notna()
        bin_index = bins[valid].astype(int).to_numpy()
        values[valid.to_numpy()] = self.bin_values_[bin_index]
        return clip_0_1(values)


def make_v3_pipeline(estimator):
    return Pipeline(
        steps=[
            ("features", StressFeatureEngineer(add_mean_working_cat_v3=True)),
            ("scorecard_bins", QuantileScorecardBinner(n_bins=5)),
            ("interactions", InteractionFeatureEngineer()),
            ("preprocess", make_ohe_preprocessor(V3_CATEGORICAL_COLS)),
            ("model", estimator),
        ]
    )


def get_v3_model_specs():
    common = {
        "n_estimators": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return [
        (
            "extratrees_v3_leaf1",
            make_v3_pipeline(
                ExtraTreesRegressor(min_samples_leaf=1, max_features=0.8, **common)
            ),
            "ExtraTrees v3, leaf=1, max_features=0.8",
        ),
        (
            "extratrees_v3_leaf2",
            make_v3_pipeline(
                ExtraTreesRegressor(min_samples_leaf=2, max_features=0.8, **common)
            ),
            "ExtraTrees v3, leaf=2, max_features=0.8",
        ),
        (
            "extratrees_v3_fullfeat",
            make_v3_pipeline(
                ExtraTreesRegressor(min_samples_leaf=2, max_features=1.0, **common)
            ),
            "ExtraTrees v3, leaf=2, max_features=1.0",
        ),
        (
            "extratrees_v3_sqrt",
            make_v3_pipeline(
                ExtraTreesRegressor(min_samples_leaf=2, max_features="sqrt", **common)
            ),
            "ExtraTrees v3, leaf=2, max_features=sqrt",
        ),
    ]


def calibration_specs():
    return [
        ("none", NoOpCalibrator()),
        ("clip_0_1", ClipCalibrator()),
        ("huber", HuberPredictionCalibrator()),
        ("isotonic", IsotonicPredictionCalibrator()),
        ("qbin_median_20", QuantileBinMedianCalibrator(n_bins=20)),
        ("qbin_median_30", QuantileBinMedianCalibrator(n_bins=30)),
    ]


def prediction_distribution(prefix, values):
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=1)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
    }


def fold_safe_calibration(raw_oof_df):
    y = raw_oof_df[TARGET].to_numpy()
    pred = raw_oof_df["oof_pred"].to_numpy()
    folds = raw_oof_df["fold"].to_numpy()

    results = []
    calibrated_oofs = []

    for cal_name, calibrator in calibration_specs():
        calibrated_pred = np.zeros(len(raw_oof_df), dtype=float)
        fold_rows = []

        for fold in sorted(raw_oof_df["fold"].unique()):
            train_mask = folds != fold
            valid_mask = folds == fold
            fold_calibrator = clone(calibrator)
            fold_calibrator.fit(pred[train_mask], y[train_mask])
            fold_pred = fold_calibrator.predict(pred[valid_mask])
            calibrated_pred[valid_mask] = fold_pred
            fold_rows.append(
                {
                    "calibration": cal_name,
                    "fold": fold,
                    "mae": mean_absolute_error(y[valid_mask], fold_pred),
                }
            )

        round2_pred = clip_round_2(calibrated_pred)
        fold_df = pd.DataFrame(fold_rows)
        results.append(
            {
                "calibration": cal_name,
                "mean_mae": float(fold_df["mae"].mean()),
                "std_mae": float(fold_df["mae"].std()),
                "mae_round2": float(mean_absolute_error(y, round2_pred)),
                "pred_std_oof": float(np.std(calibrated_pred, ddof=1)),
                "pred_min_oof": float(np.min(calibrated_pred)),
                "pred_max_oof": float(np.max(calibrated_pred)),
            }
        )

        cal_oof = raw_oof_df[[ID_COL, TARGET, "fold"]].copy()
        cal_oof["model"] = raw_oof_df["model"].iloc[0]
        cal_oof["postprocess"] = cal_name
        cal_oof["oof_pred"] = calibrated_pred
        calibrated_oofs.append(cal_oof[[ID_COL, TARGET, "model", "postprocess", "oof_pred", "fold"]])

    return pd.DataFrame(results).sort_values("mean_mae").reset_index(drop=True), pd.concat(
        calibrated_oofs, ignore_index=True
    )


def fit_full_calibrator(calibration_name, raw_pred, y):
    calibrators = dict(calibration_specs())
    calibrator = clone(calibrators[calibration_name])
    calibrator.fit(raw_pred, y)
    return calibrator


def append_v3_experiment_log(results_df):
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


def print_target_bin_mae(train_df, before_pred, after_pred, n_bins=10):
    bins = pd.qcut(train_df[TARGET], q=n_bins, labels=False, duplicates="drop")
    diag = pd.DataFrame(
        {
            "target_bin": bins,
            TARGET: train_df[TARGET],
            "mae_before": np.abs(train_df[TARGET].to_numpy() - before_pred),
            "mae_after": np.abs(train_df[TARGET].to_numpy() - after_pred),
        }
    )
    summary = (
        diag.groupby("target_bin", as_index=False)
        .agg(
            target_min=(TARGET, "min"),
            target_max=(TARGET, "max"),
            mae_before=("mae_before", "mean"),
            mae_after=("mae_after", "mean"),
        )
        .reset_index(drop=True)
    )
    print("\n=== Target-bin MAE before/after calibration ===")
    print(summary.round(6).to_string(index=False))
    return summary


def run_v3_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    all_fold_results = []
    all_summary_results = []
    all_base_oof_results = []
    notes_by_model = {}
    model_specs = get_v3_model_specs()

    for model_name, pipeline, notes in model_specs:
        print(f"\n[{model_name}]")
        fold_df, summary_df, oof_df = evaluate_pipeline_with_oof(model_name, pipeline, train_df)
        print(fold_df.pivot(index="fold", columns="postprocess", values="mae").round(6))
        print(summary_df.round(6))
        all_fold_results.append(fold_df)
        all_summary_results.append(summary_df)
        all_base_oof_results.append(oof_df)
        notes_by_model[model_name] = notes

    base_results = (
        pd.concat(all_summary_results, ignore_index=True)
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    base_oof = pd.concat(all_base_oof_results, ignore_index=True)
    best_base_model = base_results.iloc[0]["model"]
    raw_oof = base_oof[
        (base_oof["model"] == best_base_model) & (base_oof["postprocess"] == "raw")
    ].copy()

    calibration_results, calibrated_oof = fold_safe_calibration(raw_oof)
    calibration_results.insert(0, "model", best_base_model)
    calibration_results.insert(0, "feature_version", FEATURE_VERSION)
    calibration_results.insert(0, "exp_id", EXP_ID)
    calibration_results["postprocess"] = calibration_results["calibration"]
    calibration_results["notes"] = (
        notes_by_model[best_base_model] + "; fold-safe OOF calibration"
    )

    best_cal = calibration_results.iloc[0]
    best_calibration = best_cal["calibration"]

    oof_output = pd.concat([base_oof, calibrated_oof], ignore_index=True)
    oof_path = EXPERIMENT_LOG_PATH.parent / "oof_predictions_v3.csv"
    cal_path = EXPERIMENT_LOG_PATH.parent / "calibration_results_v3.csv"
    oof_output[[ID_COL, TARGET, "model", "postprocess", "oof_pred", "fold"]].to_csv(
        oof_path, index=False
    )
    calibration_results.to_csv(cal_path, index=False)

    best_pipeline = clone({name: pipe for name, pipe, _ in model_specs}[best_base_model])
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].to_numpy()
    best_pipeline.fit(X_train, y_train)
    raw_test_pred = best_pipeline.predict(test_df)

    full_calibrator = fit_full_calibrator(
        best_calibration,
        raw_oof["oof_pred"].to_numpy(),
        raw_oof[TARGET].to_numpy(),
    )
    submission_pred = clip_0_1(full_calibrator.predict(raw_test_pred))

    submission = sample_submission.copy()
    submission[TARGET] = submission_pred
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSIONS_DIR / f"{EXP_ID}_best_{best_base_model}_{best_calibration}.csv"
    submission.to_csv(submission_path, index=False)

    best_cal_oof = calibrated_oof[
        (calibrated_oof["model"] == best_base_model)
        & (calibrated_oof["postprocess"] == best_calibration)
    ].sort_values(ID_COL)
    raw_oof_sorted = raw_oof.sort_values(ID_COL)

    round2_path = None
    if best_cal["mae_round2"] < best_cal["mean_mae"]:
        round2_log_row = best_cal.copy()
        round2_log_row["calibration"] = f"{best_calibration}_round2"
        round2_log_row["postprocess"] = f"{best_calibration}_round2"
        round2_log_row["mean_mae"] = best_cal["mae_round2"]
        round2_log_row["pred_std_oof"] = float(
            np.std(clip_round_2(best_cal_oof["oof_pred"]), ddof=1)
        )
        round2_log_row["pred_min_oof"] = float(np.min(clip_round_2(best_cal_oof["oof_pred"])))
        round2_log_row["pred_max_oof"] = float(np.max(clip_round_2(best_cal_oof["oof_pred"])))
        calibration_results = pd.concat(
            [calibration_results, pd.DataFrame([round2_log_row])], ignore_index=True
        ).sort_values("mean_mae").reset_index(drop=True)

        round2_submission = sample_submission.copy()
        round2_submission[TARGET] = clip_round_2(submission_pred)
        round2_path = (
            SUBMISSIONS_DIR
            / f"{EXP_ID}_best_{best_base_model}_{best_calibration}_round2.csv"
        )
        round2_submission.to_csv(round2_path, index=False)

    calibrated_submission_path = None
    nontrivial = calibration_results[
        ~calibration_results["calibration"].isin(["none", "clip_0_1"])
        & ~calibration_results["calibration"].str.endswith("_round2")
    ].sort_values("mean_mae")
    if not nontrivial.empty:
        best_nontrivial = nontrivial.iloc[0]
        nontrivial_calibrator = fit_full_calibrator(
            best_nontrivial["calibration"],
            raw_oof["oof_pred"].to_numpy(),
            raw_oof[TARGET].to_numpy(),
        )
        calibrated_submission = sample_submission.copy()
        calibrated_submission[TARGET] = clip_0_1(
            nontrivial_calibrator.predict(raw_test_pred)
        )
        calibrated_submission_path = (
            SUBMISSIONS_DIR
            / f"{EXP_ID}_best_calibrated_{best_base_model}_{best_nontrivial['calibration']}.csv"
        )
        calibrated_submission.to_csv(calibrated_submission_path, index=False)

    print("\n=== V3 calibration comparison ===")
    print(
        calibration_results[
            [
                "model",
                "calibration",
                "mean_mae",
                "std_mae",
                "mae_round2",
                "pred_std_oof",
                "pred_min_oof",
                "pred_max_oof",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("\n=== V3 prediction distribution diagnostics ===")
    for stats in (
        prediction_distribution("train_target", train_df[TARGET]),
        prediction_distribution("base_oof_pred", raw_oof_sorted["oof_pred"]),
        prediction_distribution("calibrated_oof_pred", best_cal_oof["oof_pred"]),
        prediction_distribution("submission_pred", submission_pred),
    ):
        print(pd.Series(stats).round(6).to_string())

    print_target_bin_mae(
        train_df.sort_values(ID_COL),
        raw_oof_sorted["oof_pred"].to_numpy(),
        best_cal_oof["oof_pred"].to_numpy(),
    )

    append_v3_experiment_log(calibration_results)

    print("\n=== V3 base model comparison ===")
    print(base_results.round(6).to_string(index=False))
    print(f"\nBest base model: {best_base_model}")
    print(
        f"Best calibration: {best_calibration} / CV MAE={best_cal['mean_mae']:.6f} "
        f"/ round2 MAE={best_cal['mae_round2']:.6f}"
    )
    print(f"Saved OOF: {oof_path}")
    print(f"Saved calibration results: {cal_path}")
    print(f"Saved submission: {submission_path}")
    if round2_path:
        print(f"Saved round2 submission: {round2_path}")
    if calibrated_submission_path:
        print(f"Saved calibrated submission: {calibrated_submission_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")

    return base_results, calibration_results, oof_output, submission_path


if __name__ == "__main__":
    run_v3_experiments()
