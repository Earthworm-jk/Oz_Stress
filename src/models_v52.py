from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
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
    V3_CATEGORICAL_COLS,
)
from src.models_v3 import make_v3_pipeline
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v52_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SPLITTER = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

BASE_NUMERIC = [
    "age",
    "height",
    "weight",
    "cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "glucose",
    "bone_density",
    "mean_working",
    "bmi",
    "gender",
    "activity",
    "sleep_pattern",
    "edu_level",
]
OHE_COLS = ["smoke_status", "medical_history", "family_medical_history"]
DERIVED_BY_SET = {
    "S1_only": [],
    "S1_plus_bp": ["pulse_pressure", "map"],
    "S1_plus_glucose_cholesterol_ratio": ["glucose_cholesterol_ratio"],
    "S1_plus_cholesterol_glucose_product": ["cholesterol_glucose_product"],
    "S1_plus_all_S2": [
        "pulse_pressure",
        "map",
        "glucose_cholesterol_ratio",
        "cholesterol_glucose_product",
    ],
}


class V52FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_set="S1_plus_all_S2"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        X_df["mean_working"] = X_df["mean_working"].fillna(0)
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)

        for col in ["smoke_status", "medical_history", "family_medical_history", "edu_level"]:
            X_df[col] = X_df[col].astype("object").fillna("Unknown")

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

        X_df["pulse_pressure"] = X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])


def feature_columns(feature_set):
    return BASE_NUMERIC + DERIVED_BY_SET[feature_set], OHE_COLS


def make_v52_preprocessor(feature_set, dense=True):
    numeric_cols, ohe_cols = feature_columns(feature_set)
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
    steps = [("preprocess", preprocessor)]
    if dense:
        steps.append(("dense", DenseTransformer()))
    return Pipeline(steps)


def make_v52_pipeline(feature_set, estimator, target_mode="raw", dense=True):
    return Pipeline(
        steps=[
            ("features", V52FeatureEngineer(feature_set=feature_set)),
            ("preprocess", make_v52_preprocessor(feature_set, dense=dense)),
            ("model", TargetModeRegressor(estimator, target_mode=target_mode)),
        ]
    )


def rbf_estimator():
    return SVR(
        kernel="rbf",
        C=3.963530707518144,
        gamma=1.0631617004546035,
        epsilon=0.0,
        shrinking=True,
        cache_size=500,
    )


def evaluate_oof(train_df, pipeline, model_name, postprocess="clip_0_1_round2"):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    for fold, (tr_idx, va_idx) in enumerate(SPLITTER.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
    pred = clip_round_2(raw_oof) if postprocess == "clip_0_1_round2" else clip_0_1(raw_oof)
    fold_maes = [
        mean_absolute_error(y[folds == fold], pred[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    summary = {
        "exp_id": EXP_ID,
        "model": model_name,
        "postprocess": postprocess,
        "mean_mae": float(np.mean(fold_maes)),
        "std_mae": float(np.std(fold_maes, ddof=1)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "model": model_name,
            "postprocess": postprocess,
            "fold": folds,
            "oof_pred": pred,
        }
    )
    return summary, oof


def run_feature_ablation(train_df):
    rows = []
    for feature_set in DERIVED_BY_SET:
        pipeline = make_v52_pipeline(feature_set, rbf_estimator(), target_mode="raw", dense=True)
        summary, _ = evaluate_oof(
            train_df,
            pipeline,
            model_name=f"svr_rbf_raw_{feature_set}",
            postprocess="clip_0_1_round2",
        )
        summary["feature_set"] = feature_set
        summary["notes"] = "S2 ablation with raw target RBF SVR"
        rows.append(summary)
        print(f"ablation {feature_set}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)


def load_reference_oofs(train_df):
    v3 = pd.read_csv(EXPERIMENT_LOG_PATH.parent / "oof_predictions_v3.csv")
    v51 = pd.read_csv(EXPERIMENT_LOG_PATH.parent / "oof_predictions_v51.csv")

    v3_oof = v3[
        (v3["model"].eq("extratrees_v3_leaf1")) & (v3["postprocess"].eq("clip_0_1_round2"))
    ][[ID_COL, "oof_pred"]].rename(columns={"oof_pred": "v3_extratrees"})
    raw_oof = v51[
        (v51["model"].eq("svr_rbf_raw")) & (v51["postprocess"].eq("clip_0_1_round2"))
    ][[ID_COL, "oof_pred"]].rename(columns={"oof_pred": "rbf_raw"})
    quant_oof = v51[
        (v51["model"].eq("svr_rbf_quantile")) & (v51["postprocess"].eq("clip_0_1_round2"))
    ][[ID_COL, "oof_pred"]].rename(columns={"oof_pred": "rbf_quantile"})

    merged = train_df[[ID_COL, TARGET]].merge(v3_oof, on=ID_COL).merge(raw_oof, on=ID_COL).merge(quant_oof, on=ID_COL)
    return merged


def target_bin_mae(ref_oof, n_bins=10):
    bins = pd.qcut(ref_oof[TARGET], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for name in ["v3_extratrees", "rbf_raw", "rbf_quantile"]:
        tmp = ref_oof.copy()
        tmp["target_bin"] = bins
        tmp["abs_error"] = np.abs(tmp[TARGET] - tmp[name])
        grouped = tmp.groupby("target_bin", as_index=False).agg(
            target_min=(TARGET, "min"),
            target_max=(TARGET, "max"),
            mae=("abs_error", "mean"),
        )
        grouped.insert(0, "model", name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def prediction_distribution(ref_oof):
    rows = []
    for name, values in {
        "target": ref_oof[TARGET],
        "v3_extratrees": ref_oof["v3_extratrees"],
        "rbf_raw": ref_oof["rbf_raw"],
        "rbf_quantile": ref_oof["rbf_quantile"],
    }.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "name": name,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "q01": float(np.quantile(arr, 0.01)),
                "q05": float(np.quantile(arr, 0.05)),
                "q50": float(np.quantile(arr, 0.50)),
                "q95": float(np.quantile(arr, 0.95)),
                "q99": float(np.quantile(arr, 0.99)),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbor_diagnostic(train_df):
    specs = [
        ("ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ("linearsvr", LinearSVR(C=1.0, epsilon=0.0, random_state=RANDOM_STATE, max_iter=20000)),
        ("knn_k5", KNeighborsRegressor(n_neighbors=5, weights="uniform")),
        ("knn_k10", KNeighborsRegressor(n_neighbors=10, weights="uniform")),
        ("knn_k20", KNeighborsRegressor(n_neighbors=20, weights="uniform")),
    ]
    rows = []
    for name, estimator in specs:
        pipeline = make_v52_pipeline("S1_plus_all_S2", estimator, target_mode="raw", dense=True)
        summary, _ = evaluate_oof(train_df, pipeline, model_name=name, postprocess="clip_0_1")
        summary["diagnostic"] = name
        summary["notes"] = "RobustScaler S2 representation; 10-fold OOF"
        rows.append(summary)
        print(f"nn diagnostic {name}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae").reset_index(drop=True)


def final_touch_candidates(ref_oof):
    candidates = {
        "raw_rbf": ref_oof["rbf_raw"],
        "quantile_rbf": ref_oof["rbf_quantile"],
        "blend_0p9_raw_0p1_quantile": 0.9 * ref_oof["rbf_raw"] + 0.1 * ref_oof["rbf_quantile"],
        "blend_0p8_raw_0p2_quantile": 0.8 * ref_oof["rbf_raw"] + 0.2 * ref_oof["rbf_quantile"],
        "blend_0p9_raw_0p1_v3": 0.9 * ref_oof["rbf_raw"] + 0.1 * ref_oof["v3_extratrees"],
    }
    rows = []
    y = ref_oof[TARGET].to_numpy()
    for name, values in candidates.items():
        pred = clip_round_2(values)
        rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "postprocess": "clip_0_1_round2",
                "mae": float(mean_absolute_error(y, pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "notes": "OOF final touch candidate; no LB-based tuning",
            }
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def fit_predict_v52(train_df, test_df, target_mode):
    pipeline = make_v52_pipeline("S1_plus_all_S2", rbf_estimator(), target_mode=target_mode, dense=True)
    pipeline.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    return clip_0_1(pipeline.predict(test_df))


def fit_predict_v3(train_df, test_df):
    from sklearn.ensemble import ExtraTreesRegressor

    model = make_v3_pipeline(
        ExtraTreesRegressor(
            n_estimators=1000,
            min_samples_leaf=1,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
    return clip_0_1(model.predict(test_df))


def save_top_submissions(train_df, test_df, sample_submission, final_df):
    raw_pred = fit_predict_v52(train_df, test_df, target_mode="raw")
    quant_pred = fit_predict_v52(train_df, test_df, target_mode="quantile_target")
    v3_pred = None
    pred_lookup = {
        "raw_rbf": raw_pred,
        "quantile_rbf": quant_pred,
        "blend_0p9_raw_0p1_quantile": 0.9 * raw_pred + 0.1 * quant_pred,
        "blend_0p8_raw_0p2_quantile": 0.8 * raw_pred + 0.2 * quant_pred,
    }

    paths = []
    for _, row in final_df.head(2).iterrows():
        candidate = row["candidate"]
        if candidate == "blend_0p9_raw_0p1_v3" and v3_pred is None:
            v3_pred = fit_predict_v3(train_df, test_df)
            pred_lookup[candidate] = 0.9 * raw_pred + 0.1 * v3_pred
        pred = clip_round_2(pred_lookup[candidate])
        submission = sample_submission.copy()
        submission[TARGET] = pred
        path = SUBMISSIONS_DIR / f"{EXP_ID}_{candidate}.csv"
        submission.to_csv(path, index=False)
        paths.append(path)
    return paths


def run_v52_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    reports_dir = EXPERIMENT_LOG_PATH.parent
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    ablation = run_feature_ablation(train_df)
    ref_oof = load_reference_oofs(train_df)
    binwise = target_bin_mae(ref_oof)
    dist = prediction_distribution(ref_oof)
    nn_diag = nearest_neighbor_diagnostic(train_df)
    final_df = final_touch_candidates(ref_oof)

    ablation.to_csv(reports_dir / "v52_feature_ablation.csv", index=False)
    binwise.to_csv(reports_dir / "v52_binwise_mae.csv", index=False)
    dist.to_csv(reports_dir / "v52_prediction_distribution.csv", index=False)
    nn_diag.to_csv(reports_dir / "v52_nearest_neighbor_diagnostic.csv", index=False)
    final_df.to_csv(reports_dir / "v52_final_touch_candidates.csv", index=False)
    paths = save_top_submissions(train_df, test_df, sample_submission, final_df)

    print("\n=== V5.2 feature ablation ===")
    print(ablation[["feature_set", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== V5.2 prediction distribution ===")
    print(dist.round(6).to_string(index=False))
    print("\n=== V5.2 nearest-neighbor diagnostic ===")
    print(nn_diag[["diagnostic", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== V5.2 final touch candidates ===")
    print(final_df.round(6).to_string(index=False))
    for path in paths:
        print(f"Saved submission: {path}")
    return ablation, binwise, dist, nn_diag, final_df, paths


if __name__ == "__main__":
    run_v52_experiments()
