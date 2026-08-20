from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
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
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
V3_CV_MAE = 0.179860
V3_LB_MAE = 0.16743

NUMERIC_BASE = [
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
]
ORDINAL_COLS = ["gender", "activity", "sleep_pattern", "edu_level"]
OHE_BASE = ["smoke_status", "medical_history", "family_medical_history"]
S2_DERIVED = [
    "pulse_pressure",
    "map",
    "glucose_cholesterol_ratio",
    "cholesterol_glucose_product",
]
S3_EXTRA = ["mean_working_missing"]
S3_OHE = ["mean_working_cat_v3"]

POSTPROCESSORS = {
    "raw": lambda x: np.asarray(x, dtype=float),
    "clip_0_1": clip_0_1,
    "clip_0_1_round2": clip_round_2,
}


class V5ScoreFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_set="S1_simple_score_encoding"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        X_df["mean_working"] = X_df["mean_working"].fillna(0)

        height_m = X_df["height"] / 100.0
        X_df["bmi"] = X_df["weight"] / np.square(height_m)

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

        if self.feature_set in {"S2_core_derived", "S3_mean_working_score"}:
            X_df["pulse_pressure"] = (
                X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
            )
            X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
            X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
            X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]

        if self.feature_set == "S3_mean_working_score":
            original_mean_working = X["mean_working"]
            X_df["mean_working_missing"] = original_mean_working.isna().astype("int8")
            X_df["mean_working_cat_v3"] = self._mean_working_category_v3(original_mean_working)

        drop_cols = [ID_COL, TARGET]
        return X_df.drop(columns=[c for c in drop_cols if c in X_df.columns])

    @staticmethod
    def _mean_working_category_v3(values):
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


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


class TargetModeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, target_mode="raw"):
        self.estimator = estimator
        self.target_mode = target_mode

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        y_arr = np.asarray(y, dtype=float)
        if self.target_mode == "raw":
            self.transformer_ = None
            y_fit = y_arr
        elif self.target_mode == "y100":
            self.transformer_ = None
            y_fit = y_arr * 100.0
        elif self.target_mode == "quantile_target":
            n_quantiles = min(1000, len(y_arr))
            self.transformer_ = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=RANDOM_STATE,
            )
            y_fit = self.transformer_.fit_transform(y_arr.reshape(-1, 1)).ravel()
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")
        self.estimator_.fit(X, y_fit)
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X), dtype=float)
        if self.target_mode == "y100":
            return pred / 100.0
        if self.target_mode == "quantile_target":
            return self.transformer_.inverse_transform(pred.reshape(-1, 1)).ravel()
        return pred


def feature_columns(feature_set):
    numeric_cols = NUMERIC_BASE.copy()
    ohe_cols = OHE_BASE.copy()
    if feature_set in {"S2_core_derived", "S3_mean_working_score"}:
        numeric_cols += S2_DERIVED
    if feature_set == "S3_mean_working_score":
        numeric_cols += S3_EXTRA
        ohe_cols += S3_OHE
    numeric_cols += ORDINAL_COLS
    return numeric_cols, ohe_cols


def make_preprocessor(feature_set, scale=False, dense=False):
    numeric_cols, ohe_cols = feature_columns(feature_set)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", RobustScaler()))

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
    steps = [("preprocess", preprocessor)]
    if dense:
        steps.append(("dense", DenseTransformer()))
    return Pipeline(steps)


def make_pipeline(feature_set, estimator, target_mode, scale=False, dense=False):
    return Pipeline(
        steps=[
            ("features", V5ScoreFeatureEngineer(feature_set=feature_set)),
            ("preprocess", make_preprocessor(feature_set, scale=scale, dense=dense)),
            ("model", TargetModeRegressor(estimator, target_mode=target_mode)),
        ]
    )


def available_model_specs():
    specs = [
        ("A_linear_scorecard", "ridge", Ridge(alpha=10.0, random_state=RANDOM_STATE), True, False),
        (
            "A_linear_scorecard",
            "elasticnet",
            ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=20000, random_state=RANDOM_STATE),
            True,
            False,
        ),
        (
            "A_linear_scorecard",
            "linearsvr",
            LinearSVR(C=1.0, epsilon=0.0, random_state=RANDOM_STATE, max_iter=10000),
            True,
            True,
        ),
        (
            "B_binned_or_tree_scorecard",
            "extratrees_v3_setting",
            ExtraTreesRegressor(
                n_estimators=1000,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        (
            "B_binned_or_tree_scorecard",
            "randomforest",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        (
            "D_kernel_latent_score",
            "knn",
            KNeighborsRegressor(n_neighbors=35, weights="distance"),
            True,
            True,
        ),
        (
            "D_kernel_latent_score",
            "kernelridge_rbf",
            KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1),
            True,
            True,
        ),
        (
            "D_kernel_latent_score",
            "svr_rbf",
            SVR(
                kernel="rbf",
                C=3.963530707518144,
                gamma=1.0631617004546035,
                epsilon=0.0,
                shrinking=True,
                cache_size=500,
            ),
            True,
            True,
        ),
    ]

    try:
        from xgboost import XGBRegressor

        specs.append(
            (
                "C_boosting",
                "xgboost",
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
                ),
                False,
                False,
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor

        specs.append(
            (
                "C_boosting",
                "lightgbm",
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
                ),
                False,
                False,
            )
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostRegressor

        specs.append(
            (
                "C_boosting",
                "catboost",
                CatBoostRegressor(
                    loss_function="MAE",
                    eval_metric="MAE",
                    iterations=800,
                    learning_rate=0.03,
                    depth=6,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                ),
                False,
                False,
            )
        )
    except ImportError:
        pass

    return specs


def make_splitter(y, cv_type="kfold", n_splits=10):
    if cv_type == "stratified5":
        bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return splitter.split(np.zeros(len(y)), bins)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return splitter.split(np.zeros(len(y)))


def evaluate_spec(train_df, spec, feature_set, target_mode, cv_type="kfold"):
    hypothesis_group, model_name, estimator, scale, dense = spec
    pipeline = make_pipeline(feature_set, estimator, target_mode, scale=scale, dense=dense)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(make_splitter(y, cv_type=cv_type), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold

    rows = []
    oof_rows = []
    for post_name, post_func in POSTPROCESSORS.items():
        pred = post_func(raw_oof)
        fold_maes = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fold_maes.append(mean_absolute_error(y[mask], pred[mask]))
            oof_rows.append(
                pd.DataFrame(
                    {
                        ID_COL: train_df.loc[mask, ID_COL].to_numpy(),
                        TARGET: train_df.loc[mask, TARGET].to_numpy(),
                        "exp_id": EXP_ID,
                        "model": model_name,
                        "feature_set": feature_set,
                        "target_mode": target_mode,
                        "postprocess": post_name,
                        "fold": int(fold),
                        "oof_pred": pred[mask],
                    }
                )
            )
        rows.append(
            {
                "exp_id": EXP_ID,
                "hypothesis_group": hypothesis_group,
                "feature_set": feature_set,
                "model": model_name,
                "target_mode": target_mode,
                "postprocess": post_name,
                "mean_mae": float(np.mean(fold_maes)),
                "std_mae": float(np.std(fold_maes, ddof=1)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "notes": f"{cv_type}; {'scaled' if scale else 'unscaled'}; {'dense' if dense else 'sparse-ok'}",
            }
        )
    return pd.DataFrame(rows), pd.concat(oof_rows, ignore_index=True), pipeline


def default_experiment_plan():
    feature_sets = [
        "S1_simple_score_encoding",
        "S2_core_derived",
        "S3_mean_working_score",
    ]
    target_modes_by_model = {
        "ridge": ["raw", "y100", "quantile_target"],
        "elasticnet": ["raw", "y100"],
        "linearsvr": ["raw", "y100"],
        "extratrees_v3_setting": ["raw", "y100"],
        "randomforest": ["raw"],
        "xgboost": ["raw"],
        "lightgbm": ["raw"],
        "catboost": ["raw"],
        "knn": ["raw"],
        "kernelridge_rbf": ["raw", "quantile_target"],
        "svr_rbf": ["raw", "y100", "quantile_target"],
    }
    return feature_sets, target_modes_by_model


def should_run_default_combo(feature_set, model_name, target_mode):
    core_all_feature_sets = {
        ("ridge", "raw"),
        ("ridge", "quantile_target"),
        ("extratrees_v3_setting", "raw"),
        ("svr_rbf", "raw"),
        ("svr_rbf", "quantile_target"),
    }
    if (model_name, target_mode) in core_all_feature_sets:
        return True

    if feature_set != "S3_mean_working_score":
        return False

    s3_model_screen = {
        ("elasticnet", "raw"),
        ("linearsvr", "raw"),
        ("randomforest", "raw"),
        ("xgboost", "raw"),
        ("lightgbm", "raw"),
        ("catboost", "raw"),
        ("knn", "raw"),
        ("kernelridge_rbf", "raw"),
        ("kernelridge_rbf", "quantile_target"),
        ("svr_rbf", "y100"),
        ("extratrees_v3_setting", "y100"),
    }
    return (model_name, target_mode) in s3_model_screen


def append_experiment_log(comparison_df):
    log_df = comparison_df.rename(
        columns={
            "hypothesis_group": "feature_version",
            "pred_std": "pred_std_oof",
            "pred_min": "pred_min_oof",
            "pred_max": "pred_max_oof",
        }
    )
    log_df["notes"] = (
        log_df["notes"]
        + "; feature_set="
        + log_df["feature_set"]
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


def fit_predict_submission(train_df, test_df, sample_submission, result_row, specs):
    spec = next(s for s in specs if s[1] == result_row["model"])
    pipeline = make_pipeline(
        result_row["feature_set"],
        spec[2],
        result_row["target_mode"],
        scale=spec[3],
        dense=spec[4],
    )
    pipeline.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    pred = POSTPROCESSORS[result_row["postprocess"]](pipeline.predict(test_df))
    submission = sample_submission.copy()
    submission[TARGET] = pred
    return submission


def print_interpretation(comparison):
    best = comparison.iloc[0]
    best_group = best["hypothesis_group"]
    print("\n=== V5 interpretation ===")
    print(f"Best local CV: {best['model']} / {best['feature_set']} / {best['target_mode']} / {best['postprocess']}")
    print(f"Best CV MAE: {best['mean_mae']:.6f}")
    print(f"V3 CV MAE: {V3_CV_MAE:.6f}, V3 LB MAE: {V3_LB_MAE:.5f}")
    if best_group == "D_kernel_latent_score":
        print("Kernel/SVR 계열이 가장 강하면 부드러운 비선형 latent stress scale 가설에 힘이 실립니다.")
    elif best_group == "B_binned_or_tree_scorecard":
        print("Tree 계열이 가장 강하면 구간형 scorecard 가설을 유지하는 쪽이 자연스럽습니다.")
    elif best_group == "A_linear_scorecard":
        print("선형 계열이 가장 강하면 단순 가산식 scorecard 가능성이 커집니다.")
    else:
        print("Boosting 계열이 가장 강하면 구간성과 부드러운 비선형성이 섞인 함수일 가능성이 있습니다.")


def run_v5_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    feature_sets, target_modes_by_model = default_experiment_plan()
    specs = available_model_specs()
    all_results = []
    all_oof = []

    for feature_set in feature_sets:
        for spec in specs:
            model_name = spec[1]
            target_modes = target_modes_by_model.get(model_name, ["raw"])
            for target_mode in target_modes:
                if not should_run_default_combo(feature_set, model_name, target_mode):
                    continue
                print(f"[{feature_set}] {model_name} / {target_mode}")
                result_df, oof_df, _ = evaluate_spec(
                    train_df, spec, feature_set, target_mode, cv_type="kfold"
                )
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
    comparison_path = reports_dir / "v5_score_hypothesis_model_comparison.csv"
    oof_path = reports_dir / "oof_predictions_v5.csv"
    comparison.to_csv(comparison_path, index=False)
    oof_output[
        [ID_COL, TARGET, "exp_id", "model", "feature_set", "target_mode", "postprocess", "fold", "oof_pred"]
    ].to_csv(oof_path, index=False)
    append_experiment_log(comparison)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    best = comparison.iloc[0]
    best_submission = fit_predict_submission(train_df, test_df, sample_submission, best, specs)
    best_path = (
        SUBMISSIONS_DIR
        / f"{EXP_ID}_best_{best['model']}_{best['feature_set']}_{best['target_mode']}_{best['postprocess']}.csv"
    )
    best_submission.to_csv(best_path, index=False)

    svr_rows = comparison[comparison["model"].eq("svr_rbf")]
    svr_path = None
    if not svr_rows.empty:
        best_svr = svr_rows.iloc[0]
        if not (
            best_svr["model"] == best["model"]
            and best_svr["feature_set"] == best["feature_set"]
            and best_svr["target_mode"] == best["target_mode"]
            and best_svr["postprocess"] == best["postprocess"]
        ):
            svr_submission = fit_predict_submission(train_df, test_df, sample_submission, best_svr, specs)
            svr_path = (
                SUBMISSIONS_DIR
                / f"{EXP_ID}_best_svr_{best_svr['feature_set']}_{best_svr['target_mode']}_{best_svr['postprocess']}.csv"
            )
            svr_submission.to_csv(svr_path, index=False)

    print("\n=== V5 top results ===")
    print(
        comparison[
            [
                "hypothesis_group",
                "feature_set",
                "model",
                "target_mode",
                "postprocess",
                "mean_mae",
                "std_mae",
                "pred_std",
                "pred_min",
                "pred_max",
            ]
        ]
        .head(20)
        .round(6)
        .to_string(index=False)
    )
    print_interpretation(comparison)
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved OOF: {oof_path}")
    print(f"Saved best submission: {best_path}")
    if svr_path:
        print(f"Saved best SVR submission: {svr_path}")
    print(f"Updated experiment log: {EXPERIMENT_LOG_PATH}")
    return comparison, oof_output, best_path


if __name__ == "__main__":
    run_v5_experiments()
