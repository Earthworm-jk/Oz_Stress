from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVR

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
from src.models_v5 import DenseTransformer
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v53_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASELINE_RAW_RBF_CV = 0.139413
SPLITTER = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def rbf_estimator():
    return SVR(
        kernel="rbf",
        C=3.963530707518144,
        gamma=1.0631617004546035,
        epsilon=0.0,
        shrinking=True,
        cache_size=500,
    )


class V53FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_variant="ordinal_current",
        mean_working_variant="zero",
        body_variant="height_weight_bmi",
        metabolic_variant="all_s2",
    ):
        self.categorical_variant = categorical_variant
        self.mean_working_variant = mean_working_variant
        self.body_variant = body_variant
        self.metabolic_variant = metabolic_variant

    def fit(self, X, y=None):
        X_df = X.copy()
        self.mean_working_median_ = float(X_df["mean_working"].median())
        return self

    def transform(self, X):
        X_df = X.copy()
        for col in ["smoke_status", "medical_history", "family_medical_history", "edu_level"]:
            X_df[col] = X_df[col].astype("object").fillna("Unknown")

        mean_working = X_df["mean_working"]
        X_df["mean_working_missing"] = mean_working.isna().astype("int8")
        if self.mean_working_variant == "zero":
            X_df["mean_working"] = mean_working.fillna(0)
        elif self.mean_working_variant == "minus1":
            X_df["mean_working"] = mean_working.fillna(-1)
        elif self.mean_working_variant == "median_flag":
            X_df["mean_working"] = mean_working.fillna(self.mean_working_median_)
        elif self.mean_working_variant == "sentinel99":
            X_df["mean_working"] = mean_working.fillna(99)
        elif self.mean_working_variant == "zero_flag":
            X_df["mean_working"] = mean_working.fillna(0)
        else:
            raise ValueError(f"Unknown mean_working_variant: {self.mean_working_variant}")

        X_df["mean_working_group"] = self._mean_working_group(mean_working)
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)

        X_df["pulse_pressure"] = X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_ratio"] = X_df["cholesterol"] / X_df["glucose"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]
        X_df["log_cholesterol_glucose_product"] = np.log1p(X_df["cholesterol_glucose_product"])

        self._apply_categorical_encoding(X_df)
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])

    def _apply_categorical_encoding(self, X_df):
        X_df["gender"] = X_df["gender"].map({"F": 0, "M": 1}).astype(float)
        if self.categorical_variant == "ordinal_current":
            X_df["activity"] = X_df["activity"].map({"light": 0, "moderate": 1, "intense": 2}).astype(float)
            X_df["sleep_pattern"] = X_df["sleep_pattern"].map(
                {"sleep difficulty": 0, "normal": 1, "oversleeping": 2}
            ).astype(float)
            X_df["edu_level"] = X_df["edu_level"].map(
                {
                    "Unknown": 0,
                    "high school diploma": 1,
                    "bachelors degree": 2,
                    "graduate degree": 3,
                }
            ).astype(float)
        elif self.categorical_variant == "onehot_core":
            pass
        elif self.categorical_variant == "risk_order":
            X_df["activity"] = X_df["activity"].map({"intense": 0, "moderate": 1, "light": 2}).astype(float)
            X_df["sleep_pattern"] = X_df["sleep_pattern"].map(
                {"normal": 0, "oversleeping": 1, "sleep difficulty": 2}
            ).astype(float)
            X_df["edu_level"] = X_df["edu_level"].map(
                {
                    "Unknown": -1,
                    "high school diploma": 0,
                    "bachelors degree": 1,
                    "graduate degree": 2,
                }
            ).astype(float)
        else:
            raise ValueError(f"Unknown categorical_variant: {self.categorical_variant}")

    @staticmethod
    def _mean_working_group(values):
        return np.select(
            [
                values.isna(),
                values <= 6,
                values == 7,
                values == 8,
                values == 9,
                values == 10,
                values == 11,
                values >= 12,
            ],
            [
                "missing",
                "low_<=6",
                "work_7",
                "work_8",
                "work_9",
                "work_10",
                "work_11",
                "work_12plus",
            ],
            default="work_12plus",
        )


def columns_for_variant(config):
    body_cols = {
        "height_weight_bmi": ["height", "weight", "bmi"],
        "height_weight_only": ["height", "weight"],
        "bmi_only": ["bmi"],
        "height_bmi": ["height", "bmi"],
        "weight_bmi": ["weight", "bmi"],
    }[config["body_variant"]]

    metabolic_cols = {
        "glucose_cholesterol_only": ["glucose", "cholesterol"],
        "product": ["cholesterol_glucose_product"],
        "log_product": ["log_cholesterol_glucose_product"],
        "glucose_cholesterol_ratio": ["glucose_cholesterol_ratio"],
        "cholesterol_glucose_ratio": ["cholesterol_glucose_ratio"],
        "standardized_glucose_cholesterol": ["glucose", "cholesterol"],
        "all_s2": [
            "glucose",
            "cholesterol",
            "glucose_cholesterol_ratio",
            "cholesterol_glucose_product",
        ],
    }[config["metabolic_variant"]]

    numeric_cols = [
        "age",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "bone_density",
        "mean_working",
        "gender",
        "pulse_pressure",
        "map",
    ] + body_cols + metabolic_cols

    if config["mean_working_variant"] in {"median_flag", "zero_flag"}:
        numeric_cols.append("mean_working_missing")

    ohe_cols = ["smoke_status", "medical_history", "family_medical_history"]
    if config["categorical_variant"] == "onehot_core":
        ohe_cols += ["sleep_pattern", "activity", "edu_level"]
    else:
        numeric_cols += ["sleep_pattern", "activity", "edu_level"]

    return list(dict.fromkeys(numeric_cols)), ohe_cols


def make_pipeline(config, target_mode="raw"):
    numeric_cols, ohe_cols = columns_for_variant(config)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if config["metabolic_variant"] == "standardized_glucose_cholesterol":
        numeric_steps.append(("standard_scaler", StandardScaler()))
    numeric_steps.append(("robust_scaler", RobustScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                ohe_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    from src.models_v5 import TargetModeRegressor

    return Pipeline(
        steps=[
            ("features", V53FeatureEngineer(**config)),
            ("preprocess", preprocessor),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode=target_mode)),
        ]
    )


def experiment_configs():
    base = {
        "categorical_variant": "ordinal_current",
        "mean_working_variant": "zero",
        "body_variant": "height_weight_bmi",
        "metabolic_variant": "all_s2",
    }
    configs = [("baseline_s2_current", base)]

    for variant in ["onehot_core", "risk_order"]:
        cfg = base.copy()
        cfg["categorical_variant"] = variant
        configs.append((f"A_categorical_{variant}", cfg))

    for variant in ["minus1", "median_flag", "sentinel99", "zero_flag"]:
        cfg = base.copy()
        cfg["mean_working_variant"] = variant
        configs.append((f"B_mean_working_{variant}", cfg))

    for variant in ["height_weight_only", "bmi_only", "height_bmi", "weight_bmi"]:
        cfg = base.copy()
        cfg["body_variant"] = variant
        configs.append((f"C_body_{variant}", cfg))

    for variant in [
        "glucose_cholesterol_only",
        "product",
        "log_product",
        "glucose_cholesterol_ratio",
        "cholesterol_glucose_ratio",
        "standardized_glucose_cholesterol",
    ]:
        cfg = base.copy()
        cfg["metabolic_variant"] = variant
        configs.append((f"D_metabolic_{variant}", cfg))

    return configs


def evaluate_config(train_df, name, config, target_mode="raw"):
    pipeline = make_pipeline(config, target_mode=target_mode)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(SPLITTER.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold

    pred = clip_round_2(raw_oof)
    fold_maes = [
        mean_absolute_error(y[folds == fold], pred[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    row = {
        "exp_id": EXP_ID,
        "experiment": name,
        "target_mode": target_mode,
        "postprocess": "clip_0_1_round2",
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes, ddof=1)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        **config,
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "experiment": name,
            "target_mode": target_mode,
            "postprocess": "clip_0_1_round2",
            "fold": folds,
            "oof_pred": pred,
        }
    )
    return row, oof


def residual_analysis(train_df, best_oof):
    df = train_df.copy()
    df = df.merge(best_oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    df["y100"] = df[TARGET] * 100.0
    df["pred100"] = df["oof_pred"] * 100.0
    df["residual100"] = df["y100"] - df["pred100"]
    df["abs_residual100"] = df["residual100"].abs()
    df["mean_working_group"] = V53FeatureEngineer._mean_working_group(df["mean_working"])

    group_cols = [
        "sleep_pattern",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
        "mean_working_group",
        "edu_level",
    ]
    rows = []
    for col in group_cols:
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").fillna("Unknown")
        grouped = tmp.groupby(col, dropna=False).agg(
            count=(TARGET, "size"),
            residual_mean=("residual100", "mean"),
            residual_median=("residual100", "median"),
            mae100=("abs_residual100", "mean"),
        )
        grouped = grouped.reset_index().rename(columns={col: "group_value"})
        grouped.insert(0, "group_col", col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def append_experiment_log(comparison):
    log_df = comparison.rename(
        columns={
            "experiment": "model",
            "pred_std": "pred_std_oof",
            "pred_min": "pred_min_oof",
            "pred_max": "pred_max_oof",
        }
    )
    log_df["feature_version"] = "v53_encoding_representation"
    log_df["notes"] = (
        "raw RBF S1/S2 representation search; no LB-based calibration; "
        + "cat="
        + log_df["categorical_variant"]
        + "; mw="
        + log_df["mean_working_variant"]
        + "; body="
        + log_df["body_variant"]
        + "; metabolic="
        + log_df["metabolic_variant"]
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
    log_df = log_df[keep_cols]
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


def save_submission(train_df, test_df, sample_submission, row):
    config = {
        "categorical_variant": row["categorical_variant"],
        "mean_working_variant": row["mean_working_variant"],
        "body_variant": row["body_variant"],
        "metabolic_variant": row["metabolic_variant"],
    }
    model = make_pipeline(config, target_mode="raw")
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    pred = clip_round_2(model.predict(test_df))
    submission = sample_submission.copy()
    submission[TARGET] = pred
    path = SUBMISSIONS_DIR / f"v53_best_raw_rbf_{row['experiment']}.csv"
    submission.to_csv(path, index=False)
    return path


def run_v53_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    rows = []
    oofs = []
    for name, config in experiment_configs():
        row, oof = evaluate_config(train_df, name, config, target_mode="raw")
        rows.append(row)
        oofs.append(oof)
        print(f"{name}: {row['mean_mae']:.6f}, pred_std={row['pred_std']:.6f}")

    comparison = pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)
    oof_output = pd.concat(oofs, ignore_index=True)
    best_oof = oof_output[oof_output["experiment"].eq(comparison.iloc[0]["experiment"])].copy()
    residual = residual_analysis(train_df, best_oof)

    reports_dir = EXPERIMENT_LOG_PATH.parent
    comparison_path = reports_dir / "v53_encoding_representation_comparison.csv"
    residual_path = reports_dir / "v53_y100_residual_analysis.csv"
    oof_path = reports_dir / "oof_predictions_v53.csv"
    comparison.to_csv(comparison_path, index=False)
    residual.to_csv(residual_path, index=False)
    oof_output.to_csv(oof_path, index=False)
    append_experiment_log(comparison)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    best = comparison.iloc[0]
    if best["mean_mae"] < BASELINE_RAW_RBF_CV - 0.0001:
        paths.append(save_submission(train_df, test_df, sample_submission, best))
    else:
        print("No v5.3 submission: best CV did not clearly improve over raw RBF baseline.")

    print("\n=== V5.3 comparison ===")
    print(
        comparison[
            [
                "experiment",
                "mean_mae",
                "std_mae",
                "pred_std",
                "categorical_variant",
                "mean_working_variant",
                "body_variant",
                "metabolic_variant",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nBaseline raw RBF CV: {BASELINE_RAW_RBF_CV:.6f}")
    print(f"Best v5.3 CV: {best['mean_mae']:.6f}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved residual analysis: {residual_path}")
    print(f"Saved OOF: {oof_path}")
    for path in paths:
        print(f"Saved submission: {path}")
    return comparison, residual, oof_output, paths


if __name__ == "__main__":
    run_v53_experiments()
