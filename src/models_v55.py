from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR

from src.config import (
    ID_COL,
    PROJECT_ROOT,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.models_v54 import V54FeatureEngineer, apply_grid_postprocess, feature_columns
from src.postprocess import clip_round_2


EXP_ID = f"v55_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_CONFIG = {"mean_working_mode": "sentinel", "sentinel_value": 150.0}
BASELINE_V54_CV = 0.13488666666666665
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"


def rbf_estimator():
    return SVR(
        kernel="rbf",
        C=3.963530707518144,
        gamma=1.0631617004546035,
        epsilon=0.0,
        shrinking=True,
        cache_size=500,
    )


def make_preprocessor(config):
    numeric_cols, ohe_cols = feature_columns(config)
    return ColumnTransformer(
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


def make_rbf_pipeline(config=BASE_CONFIG):
    return Pipeline(
        steps=[
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", make_preprocessor(config)),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def load_base_oof():
    oof = pd.read_csv(REPORTS_DIR / "oof_predictions_v54.csv")
    for candidate in ["sentinel_150", "sentinel_99", "sentinel_999"]:
        subset = oof[oof["candidate"].eq(candidate)].copy()
        if not subset.empty:
            subset["oof_pred"] = clip_round_2(subset["raw_oof_pred"])
            return candidate, subset[[ID_COL, TARGET, "candidate", "fold", "raw_oof_pred", "oof_pred"]]
    raise FileNotFoundError("No v54 sentinel OOF found.")


def raw_permutation_importance(train_df, base_oof, config=BASE_CONFIG, n_sample=500):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(train_df.index.to_numpy(), size=min(n_sample, len(train_df)), replace=False)
    sample = train_df.loc[sample_idx].copy()
    y = sample[TARGET].to_numpy()

    model = make_rbf_pipeline(config)
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    base_pred = clip_round_2(model.predict(sample.drop(columns=[TARGET])))
    base_mae = mean_absolute_error(y, base_pred)

    cols = [
        "gender",
        "age",
        "height",
        "weight",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "glucose",
        "bone_density",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
        "sleep_pattern",
        "edu_level",
        "mean_working",
    ]
    rows = []
    for col in cols:
        scores = []
        for repeat in range(3):
            permuted = sample.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            pred = clip_round_2(model.predict(permuted.drop(columns=[TARGET])))
            scores.append(mean_absolute_error(y, pred) - base_mae)
        rows.append(
            {
                "exp_id": EXP_ID,
                "method": "raw_column_permutation_importance",
                "feature": col,
                "importance_mae_delta_mean": float(np.mean(scores)),
                "importance_mae_delta_std": float(np.std(scores, ddof=1)),
                "base_sample_mae": float(base_mae),
                "notes": "Direct RBF pipeline explanation fallback because shap is unavailable.",
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mae_delta_mean", ascending=False)


def make_surrogate_pipeline(config=BASE_CONFIG, random_state=RANDOM_STATE):
    return Pipeline(
        steps=[
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", make_preprocessor(config)),
            ("dense", DenseTransformer()),
            (
                "surrogate",
                ExtraTreesRegressor(
                    n_estimators=700,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def surrogate_fidelity(train_df, base_oof):
    X = train_df.drop(columns=[TARGET])
    y_sur = base_oof.sort_values(ID_COL)["oof_pred"].to_numpy()
    train_sorted = train_df.sort_values(ID_COL).reset_index(drop=True)
    X_sorted = train_sorted.drop(columns=[TARGET])

    splitter = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros(len(train_sorted), dtype=float)
    for tr_idx, va_idx in splitter.split(np.zeros(len(train_sorted))):
        model = make_surrogate_pipeline()
        model.fit(X_sorted.iloc[tr_idx], y_sur[tr_idx])
        pred[va_idx] = model.predict(X_sorted.iloc[va_idx])

    fidelity = pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "surrogate": "ExtraTreesRegressor",
                "target": "v54_rbf_oof_prediction",
                "mae_to_rbf_pred": float(mean_absolute_error(y_sur, pred)),
                "correlation": float(np.corrcoef(y_sur, pred)[0, 1]),
                "r2": float(r2_score(y_sur, pred)),
                "notes": "10-fold surrogate fidelity against RBF OOF predictions.",
            }
        ]
    )
    full_model = make_surrogate_pipeline()
    full_model.fit(X, base_oof.set_index(ID_COL).loc[train_df[ID_COL], "oof_pred"].to_numpy())
    return fidelity, full_model


def aggregate_surrogate_importance(model):
    pre = model.named_steps["preprocess"]
    names = pre.get_feature_names_out()
    importances = model.named_steps["surrogate"].feature_importances_
    known_cat = ["smoke_status", "medical_history", "family_medical_history"]
    rows = []
    for name, imp in zip(names, importances):
        raw = name.split("__", 1)[1]
        if name.startswith("cat__"):
            for cat in known_cat:
                if raw.startswith(cat + "_"):
                    raw = cat
                    break
        rows.append({"raw_feature": raw, "transformed_feature": name, "importance": imp})
    detail = pd.DataFrame(rows)
    grouped = (
        detail.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grouped.insert(0, "exp_id", EXP_ID)
    grouped["method"] = "surrogate_extra_trees_feature_importance"
    return grouped


def effect_shapes(train_df, surrogate_model):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    variables = [
        "mean_working",
        "mean_working_missing",
        "age",
        "bmi",
        "glucose",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "bone_density",
        "sleep_pattern",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
    ]
    base = train_df.copy()
    base["bmi"] = base["weight"] / np.square(base["height"] / 100.0)
    rows = []
    for var in variables:
        if var == "mean_working_missing":
            grid = [0, 1]
        elif var == "mean_working":
            grid = [np.nan, 5, 6, 7, 8, 9, 10, 11, 12, 15, 99, 150]
        elif var in ["sleep_pattern", "activity", "smoke_status", "medical_history", "family_medical_history"]:
            grid = list(train_df[var].astype("object").fillna("Unknown").drop_duplicates())
        elif var == "bmi":
            grid = np.quantile(base["bmi"], np.linspace(0.05, 0.95, 12))
        else:
            grid = np.quantile(train_df[var].dropna(), np.linspace(0.05, 0.95, 12))

        plot_x = []
        plot_y = []
        for value in grid:
            tmp = train_df.copy()
            if var == "bmi":
                current_bmi = tmp["weight"] / np.square(tmp["height"] / 100.0)
                tmp["weight"] = value * np.square(tmp["height"] / 100.0)
                display_value = float(value)
            elif var == "mean_working_missing":
                tmp["mean_working"] = np.nan if value == 1 else tmp["mean_working"].fillna(9)
                display_value = str(value)
            else:
                tmp[var] = value
                display_value = "missing" if pd.isna(value) else str(value)
            pred = surrogate_model.predict(tmp.drop(columns=[TARGET]))
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "variable": var,
                    "value": display_value,
                    "pred_mean": float(np.mean(pred)),
                    "pred_std": float(np.std(pred, ddof=1)),
                    "notes": "Surrogate PDP-style effect shape for RBF prediction.",
                }
            )
            plot_x.append(display_value)
            plot_y.append(float(np.mean(pred)))

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(plot_x)), plot_y, marker="o")
        plt.xticks(range(len(plot_x)), plot_x, rotation=45, ha="right")
        plt.ylabel("Mean surrogate prediction")
        plt.title(f"v55 PDP-style effect: {var}")
        plt.tight_layout()
        safe = var.replace("/", "_").replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"v55_pdp_ale_{safe}.png", dpi=140)
        plt.savefig(FIGURES_DIR / f"v55_shap_dependence_{safe}.png", dpi=140)
        plt.close()
    return pd.DataFrame(rows)


def residual_items(train_df, base_oof):
    df = train_df.merge(base_oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    df["y100"] = df[TARGET] * 100
    df["pred100"] = df["oof_pred"] * 100
    df["residual100"] = df["y100"] - df["pred100"]
    df["abs_residual100"] = df["residual100"].abs()
    df["mean_working_group"] = V54FeatureEngineer._mean_working_group(df["mean_working"])
    endpoint = np.select(
        [
            df[TARGET].eq(0),
            df[TARGET].between(0.01, 0.03),
            df[TARGET].between(0.97, 0.99),
            df[TARGET].eq(1),
        ],
        ["y==0", "0.01~0.03", "0.97~0.99", "y==1"],
        default="middle",
    )
    df["endpoint_group"] = endpoint
    group_cols = [
        "mean_working_group",
        "sleep_pattern",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
        "edu_level",
        "endpoint_group",
    ]
    rows = []
    for col in group_cols:
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").fillna("Unknown")
        grouped = tmp.groupby(col, as_index=False).agg(
            count=(TARGET, "size"),
            residual_mean=("residual100", "mean"),
            residual_median=("residual100", "median"),
            mae100=("abs_residual100", "mean"),
        )
        grouped = grouped.rename(columns={col: "group_value"})
        grouped.insert(0, "group_col", col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


class CandidateFeatureEngineer(V54FeatureEngineer):
    def __init__(self, candidate="baseline"):
        self.candidate = candidate
        super().__init__(mean_working_mode="sentinel", sentinel_value=150.0)

    def transform(self, X):
        X_df = super().transform(X)
        raw = X.copy()
        if self.candidate == "mean_working_missing":
            X_df["candidate_mean_working_missing"] = raw["mean_working"].isna().astype("int8")
        elif self.candidate == "mean_working_high_11plus":
            X_df["candidate_mean_working_high_11plus"] = (raw["mean_working"] >= 11).fillna(False).astype("int8")
        elif self.candidate == "mean_working_high_12plus":
            X_df["candidate_mean_working_high_12plus"] = (raw["mean_working"] >= 12).fillna(False).astype("int8")
        elif self.candidate == "sleep_risk_order_revised":
            X_df["sleep_pattern"] = raw["sleep_pattern"].map({"normal": 0, "oversleeping": 1, "sleep difficulty": 2}).astype(float)
        elif self.candidate == "activity_risk_order_revised":
            X_df["activity"] = raw["activity"].map({"intense": 0, "moderate": 1, "light": 2}).astype(float)
        elif self.candidate == "missing_and_high12":
            X_df["candidate_mean_working_missing"] = raw["mean_working"].isna().astype("int8")
            X_df["candidate_mean_working_high_12plus"] = (raw["mean_working"] >= 12).fillna(False).astype("int8")
        return X_df


def make_candidate_pipeline(candidate):
    config = BASE_CONFIG.copy()
    numeric, ohe = feature_columns(config)
    extra = []
    if candidate in ["mean_working_missing", "missing_and_high12"]:
        extra.append("candidate_mean_working_missing")
    if candidate in ["mean_working_high_11plus"]:
        extra.append("candidate_mean_working_high_11plus")
    if candidate in ["mean_working_high_12plus", "missing_and_high12"]:
        extra.append("candidate_mean_working_high_12plus")
    numeric += extra
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), ohe),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[
            ("features", CandidateFeatureEngineer(candidate=candidate)),
            ("preprocess", pre),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def evaluate_candidate_features(train_df):
    candidates = [
        "baseline",
        "mean_working_missing",
        "mean_working_high_11plus",
        "mean_working_high_12plus",
        "sleep_risk_order_revised",
        "activity_risk_order_revised",
        "missing_and_high12",
    ]
    splitter = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    rows = []
    oofs = {}
    for candidate in candidates:
        pred = np.zeros(len(train_df), dtype=float)
        for tr_idx, va_idx in splitter.split(np.zeros(len(y))):
            model = make_candidate_pipeline(candidate)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred[va_idx] = model.predict(X.iloc[va_idx])
        pred = clip_round_2(pred)
        rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": candidate,
                "mean_mae": float(mean_absolute_error(y, pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
                "notes": "Raw RBF candidate feature test against v54 sentinel150 baseline.",
            }
        )
        oofs[candidate] = pred
        print(f"candidate {candidate}: {rows[-1]['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae"), oofs


def save_submission_if_improved(train_df, test_df, sample_submission, candidate_df):
    best = candidate_df.iloc[0]
    paths = []
    if best["mean_mae"] < BASELINE_V54_CV - 0.0001 and best["candidate"] != "baseline":
        model = make_candidate_pipeline(best["candidate"])
        model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
        pred = clip_round_2(model.predict(test_df))
        sub = sample_submission.copy()
        sub[TARGET] = pred
        path = SUBMISSIONS_DIR / f"v55_best_raw_rbf_{best['candidate']}.csv"
        sub.to_csv(path, index=False)
        paths.append(path)
    return paths


def run_v55_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_candidate, base_oof = load_base_oof()
    base_oof = base_oof.sort_values(ID_COL).reset_index(drop=True)
    train_sorted = train_df.sort_values(ID_COL).reset_index(drop=True)

    direct = raw_permutation_importance(train_df, base_oof)
    fidelity, surrogate = surrogate_fidelity(train_df, base_oof)
    surrogate_importance = aggregate_surrogate_importance(surrogate)
    effects = effect_shapes(train_df, surrogate)
    residual = residual_items(train_df, base_oof)
    candidates, candidate_oofs = evaluate_candidate_features(train_df)
    submission_paths = save_submission_if_improved(train_df, test_df, sample_submission, candidates)

    direct.to_csv(REPORTS_DIR / "v55_direct_shap_importance.csv", index=False)
    fidelity.to_csv(REPORTS_DIR / "v55_surrogate_fidelity.csv", index=False)
    surrogate_importance.to_csv(REPORTS_DIR / "v55_surrogate_shap_importance.csv", index=False)
    effects.to_csv(REPORTS_DIR / "v55_effect_shapes.csv", index=False)
    residual.to_csv(REPORTS_DIR / "v55_residual_hidden_score_items.csv", index=False)
    candidates.to_csv(REPORTS_DIR / "v55_candidate_feature_test.csv", index=False)

    print("\n=== V5.5 direct importance ===")
    print(direct.head(15).round(6).to_string(index=False))
    print("\n=== V5.5 surrogate fidelity ===")
    print(fidelity.round(6).to_string(index=False))
    print("\n=== V5.5 surrogate importance ===")
    print(surrogate_importance.head(15).round(6).to_string(index=False))
    print("\n=== V5.5 candidate feature test ===")
    print(candidates.round(6).to_string(index=False))
    print("\nInterpretation summary:")
    print(f"- Base OOF source: {base_candidate}; raw RBF sentinel separation remains the reference.")
    print("- SHAP package is unavailable, so direct permutation importance and surrogate Tree importance were used.")
    print("- Candidate features are only submission-worthy if they beat the v54 sentinel baseline by a clear OOF margin.")
    if submission_paths:
        for path in submission_paths:
            print(f"Saved submission: {path}")
    else:
        print("No v5.5 submission: no candidate clearly improved over v54 sentinel baseline.")


if __name__ == "__main__":
    run_v55_experiments()
