from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR

from src.config import (
    ID_COL,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v54_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
V53_SENTINEL99_CV = 0.13491666666666666
BASE_BODY = ["height", "weight", "bmi"]
BASE_METABOLIC = ["glucose", "cholesterol", "glucose_cholesterol_ratio", "cholesterol_glucose_product"]
BASE_NUMERIC = [
    "age",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "bone_density",
    "gender",
    "activity",
    "sleep_pattern",
    "edu_level",
    "pulse_pressure",
    "map",
] + BASE_BODY + BASE_METABOLIC
BASE_OHE = ["smoke_status", "medical_history", "family_medical_history"]


def rbf_estimator():
    return SVR(
        kernel="rbf",
        C=3.963530707518144,
        gamma=1.0631617004546035,
        epsilon=0.0,
        shrinking=True,
        cache_size=500,
    )


class V54FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        mean_working_mode="sentinel",
        sentinel_value=99.0,
        add_missing_flag=False,
        missing_as_category=False,
        bin_only=False,
        tail_feature=None,
    ):
        self.mean_working_mode = mean_working_mode
        self.sentinel_value = sentinel_value
        self.add_missing_flag = add_missing_flag
        self.missing_as_category = missing_as_category
        self.bin_only = bin_only
        self.tail_feature = tail_feature

    def fit(self, X, y=None):
        self.mean_working_median_ = float(X["mean_working"].median())
        return self

    def transform(self, X):
        X_df = X.copy()
        for col in ["smoke_status", "medical_history", "family_medical_history", "edu_level"]:
            X_df[col] = X_df[col].astype("object").fillna("Unknown")

        raw_mw = X_df["mean_working"]
        missing = raw_mw.isna()
        X_df["mean_working_missing"] = missing.astype("int8")
        X_df["mean_working_group"] = self._mean_working_group(raw_mw)
        X_df["mean_working_missing_cat"] = np.where(missing, "missing", "observed")

        if self.mean_working_mode == "median":
            X_df["mean_working"] = raw_mw.fillna(self.mean_working_median_)
        elif self.mean_working_mode == "zero":
            X_df["mean_working"] = raw_mw.fillna(0)
        elif self.mean_working_mode == "sentinel":
            X_df["mean_working"] = raw_mw.fillna(self.sentinel_value)
        elif self.mean_working_mode == "keep_zero_with_category":
            X_df["mean_working"] = raw_mw.fillna(0)
        elif self.mean_working_mode == "bin_only":
            X_df["mean_working"] = raw_mw.fillna(0)
        else:
            raise ValueError(f"Unknown mean_working_mode: {self.mean_working_mode}")

        if self.tail_feature == "low_6_or_less":
            X_df["mean_working_low_6_or_less"] = (raw_mw <= 6).fillna(False).astype("int8")
        elif self.tail_feature == "high_11plus":
            X_df["mean_working_high_11plus"] = (raw_mw >= 11).fillna(False).astype("int8")
        elif self.tail_feature == "high_12plus":
            X_df["mean_working_high_12plus"] = (raw_mw >= 12).fillna(False).astype("int8")
        elif self.tail_feature == "tail_score":
            X_df["mean_working_tail_score"] = np.where(raw_mw.isna(), 0, np.maximum(raw_mw - 10, 0))

        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)
        X_df["pulse_pressure"] = X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]

        X_df["gender"] = X_df["gender"].map({"F": 0, "M": 1}).astype(float)
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
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])

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
            ["missing", "<=6", "7", "8", "9", "10", "11", ">=12"],
            default=">=12",
        )


def feature_columns(config):
    numeric = BASE_NUMERIC.copy()
    ohe = BASE_OHE.copy()

    if not config.get("bin_only", False):
        numeric.append("mean_working")
    if config.get("add_missing_flag", False):
        numeric.append("mean_working_missing")
    if config.get("missing_as_category", False):
        ohe.append("mean_working_missing_cat")
    if config.get("bin_only", False):
        ohe.append("mean_working_group")

    tail = config.get("tail_feature")
    if tail == "low_6_or_less":
        numeric.append("mean_working_low_6_or_less")
    elif tail == "high_11plus":
        numeric.append("mean_working_high_11plus")
    elif tail == "high_12plus":
        numeric.append("mean_working_high_12plus")
    elif tail == "tail_score":
        numeric.append("mean_working_tail_score")
    return numeric, ohe


def make_pipeline_from_config(config):
    numeric_cols, ohe_cols = feature_columns(config)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_cols,
            ),
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
    return Pipeline(
        steps=[
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", preprocessor),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def evaluate_config(train_df, name, config, n_splits=10, seed=RANDOM_STATE, postprocess="round2"):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    pipeline = make_pipeline_from_config(config)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold

    pred = apply_grid_postprocess(raw_oof, postprocess)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_rows.append(
            {
                "fold": int(fold),
                "mae": mean_absolute_error(y[mask], pred[mask]),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "seed": seed,
        "n_splits": n_splits,
        "postprocess": postprocess,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std()),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "candidate": name,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": pred,
        }
    )
    return summary, fold_df, oof


def apply_grid_postprocess(pred, mode):
    clipped = clip_0_1(pred)
    if mode == "clip":
        return clipped
    if mode == "round2":
        return np.round(clipped, 2)
    if mode == "floor":
        return np.floor(clipped * 100) / 100
    if mode == "ceil":
        return np.ceil(clipped * 100) / 100
    if mode.startswith("offset_"):
        offset = float(mode.replace("offset_", ""))
        return np.round(clip_0_1(clipped + offset), 2)
    raise ValueError(f"Unknown postprocess: {mode}")


def robust_scaler_stats(train_df, sentinel_value):
    observed = train_df["mean_working"].dropna()
    q1 = observed.quantile(0.25)
    q3 = observed.quantile(0.75)
    iqr = q3 - q1
    median = observed.median()
    scaled = (sentinel_value - median) / iqr if iqr else np.nan
    return median, iqr, scaled


def sentinel_sweep(train_df):
    rows = []
    folds = []
    oofs = []
    candidates = [
        ("sentinel_0", 0),
        ("sentinel_minus1", -1),
        ("median_flag", None),
        ("sentinel_16", 16),
        ("sentinel_20", 20),
        ("sentinel_30", 30),
        ("sentinel_50", 50),
        ("sentinel_99", 99),
        ("sentinel_150", 150),
        ("sentinel_999", 999),
    ]
    for name, sentinel in candidates:
        if name == "median_flag":
            config = {"mean_working_mode": "median", "add_missing_flag": True}
            sentinel_for_stats = train_df["mean_working"].median()
        else:
            config = {"mean_working_mode": "sentinel", "sentinel_value": float(sentinel)}
            sentinel_for_stats = float(sentinel)
        summary, fold_df, oof = evaluate_config(train_df, name, config)
        median, iqr, scaled = robust_scaler_stats(train_df, sentinel_for_stats)
        summary.update(
            {
                "sentinel_value": sentinel_for_stats,
                "train_mean_working_median": median,
                "train_mean_working_iqr": iqr,
                "scaled_sentinel_value": scaled,
                "notes": "sentinel sweep; fold-safe pipeline",
            }
        )
        rows.append(summary)
        fold_df.insert(0, "candidate", name)
        folds.append(fold_df)
        oofs.append(oof)
        print(f"sentinel {name}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae"), pd.concat(folds), pd.concat(oofs)


def add_sentinel_submission_stats(train_df, test_df, sentinel_df):
    config_lookup = {
        "sentinel_0": {"mean_working_mode": "sentinel", "sentinel_value": 0.0},
        "sentinel_minus1": {"mean_working_mode": "sentinel", "sentinel_value": -1.0},
        "median_flag": {"mean_working_mode": "median", "add_missing_flag": True},
        "sentinel_16": {"mean_working_mode": "sentinel", "sentinel_value": 16.0},
        "sentinel_20": {"mean_working_mode": "sentinel", "sentinel_value": 20.0},
        "sentinel_30": {"mean_working_mode": "sentinel", "sentinel_value": 30.0},
        "sentinel_50": {"mean_working_mode": "sentinel", "sentinel_value": 50.0},
        "sentinel_99": {"mean_working_mode": "sentinel", "sentinel_value": 99.0},
        "sentinel_150": {"mean_working_mode": "sentinel", "sentinel_value": 150.0},
        "sentinel_999": {"mean_working_mode": "sentinel", "sentinel_value": 999.0},
    }
    rows = []
    for _, row in sentinel_df.iterrows():
        candidate = row["candidate"]
        model = make_pipeline_from_config(config_lookup[candidate])
        model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
        pred = clip_round_2(model.predict(test_df))
        updated = row.to_dict()
        updated.update(
            {
                "submission_pred_mean": float(np.mean(pred)),
                "submission_pred_std": float(np.std(pred, ddof=1)),
                "submission_pred_min": float(np.min(pred)),
                "submission_pred_max": float(np.max(pred)),
            }
        )
        rows.append(updated)
    return pd.DataFrame(rows).sort_values("mean_mae")


def mechanism_experiment(train_df):
    configs = [
        ("mw_99_only", {"mean_working_mode": "sentinel", "sentinel_value": 99.0}),
        ("mw_99_flag", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "add_missing_flag": True}),
        ("mw_0_flag", {"mean_working_mode": "zero", "add_missing_flag": True}),
        ("mw_median_flag", {"mean_working_mode": "median", "add_missing_flag": True}),
        (
            "mw_numeric_plus_missing_category",
            {"mean_working_mode": "keep_zero_with_category", "missing_as_category": True},
        ),
        ("mw_bin_only", {"mean_working_mode": "bin_only", "bin_only": True}),
    ]
    rows = []
    oofs = []
    for name, config in configs:
        summary, _, oof = evaluate_config(train_df, name, config)
        summary["notes"] = "mean_working mechanism decomposition"
        rows.append(summary)
        oofs.append(oof)
        print(f"mechanism {name}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae"), pd.concat(oofs)


def endpoint_profile(train_df):
    df = train_df.copy()
    groups = [
        ("y_eq_0", df[TARGET].eq(0.0)),
        ("y_001_003", df[TARGET].between(0.01, 0.03)),
        ("y_004_010", df[TARGET].between(0.04, 0.10)),
        ("y_090_096", df[TARGET].between(0.90, 0.96)),
        ("y_097_099", df[TARGET].between(0.97, 0.99)),
        ("y_eq_1", df[TARGET].eq(1.0)),
    ]
    numeric_cols = [
        "age",
        "height",
        "weight",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "glucose",
        "bone_density",
        "mean_working",
    ]
    cat_cols = ["smoke_status", "medical_history", "family_medical_history", "sleep_pattern", "activity", "edu_level"]
    rows = []
    for group_name, mask in groups:
        subset = df[mask].copy()
        if subset.empty:
            continue
        row = {
            "group": group_name,
            "count": len(subset),
            "mean_working_missing_rate": float(subset["mean_working"].isna().mean()),
        }
        for col in numeric_cols:
            row[f"{col}_mean"] = float(subset[col].mean())
            row[f"{col}_median"] = float(subset[col].median())
        for col in cat_cols:
            top = subset[col].astype("object").fillna("Unknown").value_counts(normalize=True)
            row[f"{col}_top"] = top.index[0] if len(top) else ""
            row[f"{col}_top_rate"] = float(top.iloc[0]) if len(top) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def residual_analysis(train_df, oof, candidate):
    df = train_df[[ID_COL, TARGET, "mean_working", "sleep_pattern", "activity", "smoke_status", "medical_history"]].merge(
        oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left"
    )
    df["residual100"] = df[TARGET] * 100 - df["oof_pred"] * 100
    df["abs_residual100"] = df["residual100"].abs()
    df["mean_working_group"] = V54FeatureEngineer._mean_working_group(df["mean_working"])
    df["target_decile"] = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop")
    endpoint_masks = {
        "y_eq_0": df[TARGET].eq(0.0),
        "y_001_003": df[TARGET].between(0.01, 0.03),
        "y_097_099": df[TARGET].between(0.97, 0.99),
        "y_eq_1": df[TARGET].eq(1.0),
    }
    rows = []
    for name, mask in endpoint_masks.items():
        rows.append(_residual_row(candidate, "endpoint", name, df[mask]))
    for col in ["target_decile", "mean_working_group", "sleep_pattern", "activity", "smoke_status", "medical_history"]:
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").fillna("Unknown")
        for value, subset in tmp.groupby(col):
            rows.append(_residual_row(candidate, col, value, subset))
    return pd.DataFrame(rows)


def _residual_row(candidate, group_col, group_value, subset):
    return {
        "candidate": candidate,
        "group_col": group_col,
        "group_value": group_value,
        "count": len(subset),
        "residual_mean": float(subset["residual100"].mean()) if len(subset) else np.nan,
        "residual_median": float(subset["residual100"].median()) if len(subset) else np.nan,
        "mae100": float(subset["abs_residual100"].mean()) if len(subset) else np.nan,
    }


def grid_postprocess_analysis(train_df, raw_oof, candidate):
    modes = ["clip", "round2", "floor", "ceil"] + [f"offset_{x:+.3f}" for x in np.arange(-0.005, 0.0051, 0.001)]
    y = train_df[TARGET].to_numpy()
    rows = []
    for mode in modes:
        pred = apply_grid_postprocess(raw_oof, mode)
        rows.append(
            {
                "candidate": candidate,
                "postprocess": mode,
                "mae": float(mean_absolute_error(y, pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "notes": "OOF-only grid postprocess check; no LB-based offset selection",
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def tail_feature_experiment(train_df):
    configs = [
        ("tail_missing_flag", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "add_missing_flag": True}),
        ("tail_low_6_or_less", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "low_6_or_less"}),
        ("tail_high_11plus", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "high_11plus"}),
        ("tail_high_12plus", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "high_12plus"}),
        ("tail_score", {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "tail_score"}),
    ]
    rows = []
    oofs = []
    for name, config in configs:
        summary, _, oof = evaluate_config(train_df, name, config)
        summary["notes"] = "small mean_working tail feature check"
        rows.append(summary)
        oofs.append(oof)
        print(f"tail {name}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae"), pd.concat(oofs)


def stability_check(train_df, top_candidates):
    rows = []
    for candidate, config in top_candidates:
        for seed in [2024, 777]:
            summary, _, _ = evaluate_config(train_df, candidate, config, n_splits=5, seed=seed)
            summary["notes"] = "5-fold stability check on alternate seed"
            rows.append(summary)
            print(f"stability {candidate} seed={seed}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows)


def fit_predict_submission(train_df, test_df, config):
    pipeline = make_pipeline_from_config(config)
    pipeline.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    return clip_round_2(pipeline.predict(test_df))


def run_v54_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    reports_dir = TRAIN_PATH.parents[1] / "reports"
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sentinel_df, sentinel_folds, sentinel_oof = sentinel_sweep(train_df)
    sentinel_df = add_sentinel_submission_stats(train_df, test_df, sentinel_df)
    mechanism_df, mechanism_oof = mechanism_experiment(train_df)
    endpoint_df = endpoint_profile(train_df)
    tail_df, tail_oof = tail_feature_experiment(train_df)

    all_summary = pd.concat(
        [
            sentinel_df.assign(section="sentinel_sweep"),
            mechanism_df.assign(section="mechanism"),
            tail_df.assign(section="tail_features"),
        ],
        ignore_index=True,
    ).sort_values("mean_mae")
    all_oof = pd.concat([sentinel_oof, mechanism_oof, tail_oof], ignore_index=True)
    best_name = all_summary.iloc[0]["candidate"]
    best_oof = all_oof[all_oof["candidate"].eq(best_name)].copy()

    residual_df = residual_analysis(train_df, best_oof, best_name)
    v53_oof_path = reports_dir / "oof_predictions_v53.csv"
    if v53_oof_path.exists():
        v53_oof_all = pd.read_csv(v53_oof_path)
        v53_oof = v53_oof_all[
            v53_oof_all["experiment"].eq("B_mean_working_sentinel99")
        ][[ID_COL, "oof_pred"]].copy()
        residual_df = pd.concat(
            [residual_analysis(train_df, v53_oof, "v53_sentinel99"), residual_df],
            ignore_index=True,
        )
    grid_df = grid_postprocess_analysis(train_df, best_oof["raw_oof_pred"].to_numpy(), best_name)

    candidate_config_lookup = {
        "sentinel_99": {"mean_working_mode": "sentinel", "sentinel_value": 99.0},
        "sentinel_150": {"mean_working_mode": "sentinel", "sentinel_value": 150.0},
        "sentinel_999": {"mean_working_mode": "sentinel", "sentinel_value": 999.0},
        "mw_99_only": {"mean_working_mode": "sentinel", "sentinel_value": 99.0},
        "mw_99_flag": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "add_missing_flag": True},
        "tail_missing_flag": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "add_missing_flag": True},
        "tail_low_6_or_less": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "low_6_or_less"},
        "tail_high_11plus": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "high_11plus"},
        "tail_high_12plus": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "high_12plus"},
        "tail_score": {"mean_working_mode": "sentinel", "sentinel_value": 99.0, "tail_feature": "tail_score"},
    }
    top_pairs = []
    for candidate in all_summary["candidate"].head(3):
        if candidate in candidate_config_lookup:
            top_pairs.append((candidate, candidate_config_lookup[candidate]))
    stability_df = stability_check(train_df, top_pairs)

    sentinel_df.to_csv(reports_dir / "v54_mean_working_sentinel_sweep.csv", index=False)
    sentinel_folds.to_csv(reports_dir / "v54_mean_working_sentinel_sweep_folds.csv", index=False)
    mechanism_df.to_csv(reports_dir / "v54_mean_working_mechanism.csv", index=False)
    endpoint_df.to_csv(reports_dir / "v54_endpoint_profile.csv", index=False)
    residual_df.to_csv(reports_dir / "v54_endpoint_residual_analysis.csv", index=False)
    grid_df.to_csv(reports_dir / "v54_grid_postprocess.csv", index=False)
    tail_df.to_csv(reports_dir / "v54_mean_working_tail_features.csv", index=False)
    stability_df.to_csv(reports_dir / "v54_stability_check.csv", index=False)
    all_oof.to_csv(reports_dir / "oof_predictions_v54.csv", index=False)

    submission_paths = []
    stable_candidates = set(stability_df.groupby("candidate")["mean_mae"].mean().sort_values().head(2).index)
    for _, row in all_summary.head(3).iterrows():
        candidate = row["candidate"]
        if (
            row["mean_mae"] < V53_SENTINEL99_CV
            and candidate in stable_candidates
            and candidate in candidate_config_lookup
            and len(submission_paths) < 2
        ):
            pred = fit_predict_submission(train_df, test_df, candidate_config_lookup[candidate])
            submission = sample_submission.copy()
            submission[TARGET] = pred
            path = SUBMISSIONS_DIR / f"v54_best_raw_rbf_{candidate}.csv"
            submission.to_csv(path, index=False)
            submission_paths.append(path)

    print("\n=== V5.4 sentinel sweep ===")
    print(sentinel_df[["candidate", "mean_mae", "std_mae", "pred_std", "scaled_sentinel_value"]].round(6).to_string(index=False))
    print("\n=== V5.4 mechanism ===")
    print(mechanism_df[["candidate", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== V5.4 tail features ===")
    print(tail_df[["candidate", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== V5.4 grid postprocess top ===")
    print(grid_df.head(10).round(6).to_string(index=False))
    print("\n=== V5.4 stability ===")
    print(stability_df[["candidate", "seed", "n_splits", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
    print(f"\nBest v5.4 candidate: {best_name} / MAE={all_summary.iloc[0]['mean_mae']:.6f}")
    if submission_paths:
        for path in submission_paths:
            print(f"Saved submission: {path}")
    else:
        print("No v5.4 submission: candidates did not pass improvement/stability criteria.")
    return sentinel_df, mechanism_df, endpoint_df, residual_df, grid_df, tail_df, stability_df, all_oof


if __name__ == "__main__":
    run_v54_experiments()
