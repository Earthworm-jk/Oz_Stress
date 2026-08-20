from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, _tree

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
from src.models_v54 import V54FeatureEngineer, feature_columns
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v60_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_CV = 0.13491666666666666
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


def make_base_config(sentinel=99):
    return {"mean_working_mode": "sentinel", "sentinel_value": float(sentinel)}


def make_preprocessor(config, extra_numeric=None, extra_ohe=None):
    numeric, ohe = feature_columns(config)
    numeric = numeric + (extra_numeric or [])
    ohe = ohe + (extra_ohe or [])
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                ohe,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_base_pipeline(sentinel=99):
    config = make_base_config(sentinel)
    return Pipeline(
        [
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", make_preprocessor(config)),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def evaluate_oof(train_df, candidate, pipeline):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(SPLITTER.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        pred = clip_round_2(raw[va_idx])
        fold_rows.append({"candidate": candidate, "fold": fold, "mae": mean_absolute_error(y[va_idx], pred)})
    pred = clip_round_2(raw)
    summary = {
        "exp_id": EXP_ID,
        "candidate": candidate,
        "mean_mae": float(np.mean([r["mae"] for r in fold_rows])),
        "std_mae": float(np.std([r["mae"] for r in fold_rows], ddof=1)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": candidate,
            "fold": folds,
            "raw_oof_pred": raw,
            "oof_pred": pred,
        }
    )
    return summary, pd.DataFrame(fold_rows), oof


def baseline_oof(train_df):
    rows, folds, oofs = [], [], []
    for sentinel in [99, 150]:
        name = f"sentinel{sentinel}"
        summary, fold_df, oof = evaluate_oof(train_df, name, make_base_pipeline(sentinel))
        rows.append(summary)
        folds.append(fold_df)
        oofs.append(oof)
        print(f"baseline {name}: {summary['mean_mae']:.6f}")
    return pd.DataFrame(rows).sort_values("mean_mae"), pd.concat(folds), pd.concat(oofs)


def add_missing_features(df):
    out = df.copy()
    out["mean_working_missing"] = out["mean_working"].isna().astype("int8")
    out["medical_history_unknown"] = out["medical_history"].isna().astype("int8")
    out["family_medical_history_unknown"] = out["family_medical_history"].isna().astype("int8")
    out["edu_level_unknown"] = out["edu_level"].isna().astype("int8")
    flags = [
        "mean_working_missing",
        "medical_history_unknown",
        "family_medical_history_unknown",
        "edu_level_unknown",
    ]
    out["missing_count"] = out[flags].sum(axis=1)
    out["missing_pattern_code"] = (
        "MW" + out["mean_working_missing"].astype(str)
        + "_MED" + out["medical_history_unknown"].astype(str)
        + "_FAM" + out["family_medical_history_unknown"].astype(str)
        + "_EDU" + out["edu_level_unknown"].astype(str)
    )
    return out


def analysis_frame(train_df, oof):
    df = add_missing_features(train_df).merge(oof[[ID_COL, "fold", "oof_pred"]], on=ID_COL, how="left")
    df["y100"] = df[TARGET] * 100
    df["pred100"] = df["oof_pred"] * 100
    df["residual100"] = df["y100"] - df["pred100"]
    df["abs_residual100"] = df["residual100"].abs()
    df["mean_working_group"] = V54FeatureEngineer._mean_working_group(df["mean_working"])
    df["bmi"] = df["weight"] / np.square(df["height"] / 100)
    df["pulse_pressure"] = df["systolic_blood_pressure"] - df["diastolic_blood_pressure"]
    df["map"] = df["diastolic_blood_pressure"] + df["pulse_pressure"] / 3
    return df


def missing_pattern_reports(df):
    summary = (
        df.groupby("missing_pattern_code", as_index=False)
        .agg(
            count=(TARGET, "size"),
            y_mean=(TARGET, "mean"),
            y_median=(TARGET, "median"),
            y_std=(TARGET, "std"),
            missing_count=("missing_count", "first"),
        )
        .sort_values("count", ascending=False)
    )
    residual = (
        df.groupby("missing_pattern_code", as_index=False)
        .agg(
            count=(TARGET, "size"),
            y_mean=(TARGET, "mean"),
            residual_mean=("residual100", "mean"),
            residual_median=("residual100", "median"),
            residual_mae=("abs_residual100", "mean"),
        )
        .sort_values("residual_mean", key=lambda s: s.abs(), ascending=False)
    )
    residual["candidate_flag"] = np.where((residual["count"] >= 30) & (residual["residual_mean"].abs() >= 5), 1, 0)
    return summary, residual


class MissingPatternFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name=None, pattern_values=None):
        self.feature_name = feature_name
        self.pattern_values = pattern_values
        self.base = V54FeatureEngineer(mean_working_mode="sentinel", sentinel_value=99.0)

    def fit(self, X, y=None):
        self.base.fit(X, y)
        return self

    def transform(self, X):
        base = self.base.transform(X)
        helper = add_missing_features(X)
        f = self.feature_name
        if f == "missing_count":
            base["feature_missing_count"] = helper["missing_count"]
        elif f == "mw_x_edu_unknown":
            base["feature_mw_x_edu_unknown"] = helper["mean_working_missing"] * helper["edu_level_unknown"]
        elif f == "mw_x_med_unknown":
            base["feature_mw_x_med_unknown"] = helper["mean_working_missing"] * helper["medical_history_unknown"]
        elif f == "mw_x_fam_unknown":
            base["feature_mw_x_fam_unknown"] = helper["mean_working_missing"] * helper["family_medical_history_unknown"]
        elif f == "med_x_fam_unknown":
            base["feature_med_x_fam_unknown"] = helper["medical_history_unknown"] * helper["family_medical_history_unknown"]
        elif f == "all_unknown_count_high":
            base["feature_all_unknown_count_high"] = (helper["missing_count"] >= 2).astype("int8")
        elif f and f.startswith("top_pattern"):
            patterns = self.pattern_values or []
            base[f"feature_{f}"] = helper["missing_pattern_code"].isin(patterns).astype("int8")
        return base


def make_missing_feature_pipeline(feature_name, pattern_values=None):
    config = make_base_config(99)
    extra = []
    if feature_name:
        extra = [f"feature_{feature_name}"] if feature_name.startswith("top_pattern") else [f"feature_{feature_name}"]
    # map exact generated names for non-top features
    gen = {
        "missing_count": "feature_missing_count",
        "mw_x_edu_unknown": "feature_mw_x_edu_unknown",
        "mw_x_med_unknown": "feature_mw_x_med_unknown",
        "mw_x_fam_unknown": "feature_mw_x_fam_unknown",
        "med_x_fam_unknown": "feature_med_x_fam_unknown",
        "all_unknown_count_high": "feature_all_unknown_count_high",
    }
    extra = [gen.get(feature_name, f"feature_{feature_name}")] if feature_name else []
    return Pipeline(
        [
            ("features", MissingPatternFeatureEngineer(feature_name, pattern_values)),
            ("preprocess", make_preprocessor(config, extra_numeric=extra)),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def missing_feature_test(train_df, missing_residual):
    tests = [
        ("missing_count", None),
        ("mw_x_edu_unknown", None),
        ("mw_x_med_unknown", None),
        ("mw_x_fam_unknown", None),
        ("med_x_fam_unknown", None),
        ("all_unknown_count_high", None),
    ]
    top_patterns = missing_residual[missing_residual["candidate_flag"].eq(1)]["missing_pattern_code"].head(3).tolist()
    for i, pat in enumerate(top_patterns, start=1):
        tests.append((f"top_pattern_{i}", [pat]))
    rows = []
    for name, patterns in tests:
        summary, _, _ = evaluate_oof(train_df, name, make_missing_feature_pipeline(name, patterns))
        rows.append(summary)
        print(f"missing feature {name}: {summary['mean_mae']:.6f}")
    if tests:
        all_patterns = top_patterns
        summary, _, _ = evaluate_oof(train_df, "all_missing_features", make_missing_feature_pipeline("top_pattern_all", all_patterns))
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("mean_mae")


def y100_grid_diagnostic(df):
    y100 = (df[TARGET] * 100).round().astype(int)
    pred100 = (df["oof_pred"] * 100).round().astype(int)
    class_counts = y100.value_counts().sort_index().rename_axis("class_y100").reset_index(name="count")
    metrics = pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "is_integer_grid": bool(np.allclose(df[TARGET] * 100, y100)),
                "unique_classes": int(y100.nunique()),
                "exact_class_match_rate": float((y100 == pred100).mean()),
                "within_1_point_accuracy": float((np.abs(y100 - pred100) <= 1).mean()),
                "within_2_point_accuracy": float((np.abs(y100 - pred100) <= 2).mean()),
                "within_5_point_accuracy": float((np.abs(y100 - pred100) <= 5).mean()),
                "mae_points": float(np.abs(y100 - pred100).mean()),
            }
        ]
    )
    decile = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop")
    decile_err = (
        pd.DataFrame({"target_decile": decile, "abs_class_error": np.abs(y100 - pred100), "abs_error": np.abs(df[TARGET] - df["oof_pred"])})
        .groupby("target_decile", as_index=False)
        .agg(class_mae=("abs_class_error", "mean"), mae=("abs_error", "mean"))
    )
    return metrics, class_counts, decile_err


def ordinal_baseline(train_df):
    # Lightweight diagnostic classifier: predict 0..100 class then divide by 100.
    config = make_base_config(99)
    pre = Pipeline(
        [
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", make_preprocessor(config)),
            ("dense", DenseTransformer()),
        ]
    )
    X = train_df.drop(columns=[TARGET])
    y_class = (train_df[TARGET] * 100).round().astype(int).to_numpy()
    pred = np.zeros(len(train_df), dtype=float)
    for tr_idx, va_idx in SPLITTER.split(np.zeros(len(train_df))):
        model = Pipeline(
            [
                ("pre", clone(pre)),
                ("clf", ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        model.fit(X.iloc[tr_idx], y_class[tr_idx])
        proba = model.predict_proba(X.iloc[va_idx])
        classes = model.named_steps["clf"].classes_
        pred[va_idx] = (proba * classes.reshape(1, -1)).sum(axis=1) / 100.0
    pred = clip_round_2(pred)
    return pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "model": "ExtraTreesClassifier_expected_value",
                "mean_mae": float(mean_absolute_error(train_df[TARGET], pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "notes": "Ordinal/grid diagnostic only; not a submission candidate.",
            }
        ]
    )


def calibration_diagnostic(df):
    y = df[TARGET].to_numpy()
    pred = df["oof_pred"].to_numpy()
    folds = df["fold"].to_numpy()
    rows = []
    methods = ["linear", "isotonic", "piecewise_20"]
    for method in methods:
        cal_pred = np.zeros(len(df), dtype=float)
        for fold in sorted(np.unique(folds)):
            tr = folds != fold
            va = folds == fold
            if method == "linear":
                cal = LinearRegression().fit(pred[tr].reshape(-1, 1), y[tr])
                cal_pred[va] = cal.predict(pred[va].reshape(-1, 1))
            elif method == "isotonic":
                cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pred[tr], y[tr])
                cal_pred[va] = cal.predict(pred[va])
            else:
                cal_pred[va] = piecewise_predict(pred[tr], y[tr], pred[va], n_bins=20)
        for post in ["clip", "round2"]:
            pp = clip_0_1(cal_pred) if post == "clip" else clip_round_2(cal_pred)
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "calibration": method,
                    "postprocess": post,
                    "mean_mae": float(mean_absolute_error(y, pp)),
                    "pred_mean": float(np.mean(pp)),
                    "pred_std": float(np.std(pp, ddof=1)),
                    "notes": "Fold-safe OOF calibration diagnostic; not fit/evaluated on same fold.",
                }
            )
    base = df["oof_pred"].to_numpy()
    rows.append({"exp_id": EXP_ID, "calibration": "none", "postprocess": "round2", "mean_mae": float(mean_absolute_error(y, base)), "pred_mean": float(np.mean(base)), "pred_std": float(np.std(base, ddof=1)), "notes": "Baseline OOF."})
    return pd.DataFrame(rows).sort_values("mean_mae")


def piecewise_predict(train_pred, train_y, valid_pred, n_bins=20):
    edges = np.unique(np.quantile(train_pred, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = pd.cut(train_pred, edges, labels=False, include_lowest=True)
    med = pd.Series(train_y).groupby(bins).mean()
    global_mean = np.mean(train_y)
    vb = pd.cut(valid_pred, edges, labels=False, include_lowest=True)
    return np.array([med.get(int(b), global_mean) if pd.notna(b) else global_mean for b in vb])


def score_form_approximation(df):
    out_rows, rules = [], []
    y_targets = {"y100": df[TARGET] * 100, "pred100": df["oof_pred"] * 100, "residual100": df["residual100"]}
    feature_cols = ["age", "bmi", "glucose", "cholesterol", "systolic_blood_pressure", "diastolic_blood_pressure", "bone_density", "mean_working_missing"]
    X = df[feature_cols].fillna(df[feature_cols].median())
    for target_name, y in y_targets.items():
        for depth in [2, 3, 4]:
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=RANDOM_STATE)
            tree.fit(X, y)
            out_rows.append({"exp_id": EXP_ID, "target": target_name, "model": f"tree_depth_{depth}", "r2_in_sample": float(tree.score(X, y)), "notes": "Explainability approximation only."})
            for node, rule in tree_rules(tree, feature_cols).items():
                rules.append({"exp_id": EXP_ID, "target": target_name, "model": f"tree_depth_{depth}", "rule": rule})

    binned = df.copy()
    binned["age_bin"] = pd.cut(binned["age"], 5)
    binned["BMI_bin"] = pd.cut(binned["bmi"], 5)
    binned["glucose_bin"] = pd.cut(binned["glucose"], 5)
    binned["cholesterol_bin"] = pd.cut(binned["cholesterol"], 5)
    binned["BP_bin"] = pd.cut(binned["systolic_blood_pressure"], 5)
    binned["bone_density_bin"] = pd.cut(binned["bone_density"], 5)
    cat_cols = ["age_bin", "BMI_bin", "glucose_bin", "cholesterol_bin", "BP_bin", "bone_density_bin", "mean_working_group"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xb = enc.fit_transform(binned[cat_cols].astype(str))
    for target_name, y in y_targets.items():
        ridge = Ridge(alpha=10.0).fit(Xb, y)
        out_rows.append({"exp_id": EXP_ID, "target": target_name, "model": "ridge_binned_features", "r2_in_sample": float(ridge.score(Xb, y)), "notes": "Binned additive score-form approximation."})
    return pd.DataFrame(out_rows), pd.DataFrame(rules)


def tree_rules(tree, names):
    rules = {}
    t = tree.tree_

    def rec(node, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = " AND ".join(path) if path else "ALL"
            return
        name = names[t.feature[node]]
        th = t.threshold[node]
        rec(t.children_left[node], path + [f"{name} <= {th:.4f}"])
        rec(t.children_right[node], path + [f"{name} > {th:.4f}"])

    rec(0, [])
    return rules


def candidate_feature_test(train_df, missing_residual):
    feature_df = missing_feature_test(train_df, missing_residual)
    return feature_df


def submission_qa():
    candidates = {
        "v51_raw": SUBMISSIONS_DIR / "v51_svr_rbf_S2_core_derived_raw_target_clip_0_1_round2.csv",
        "v53_sentinel99": SUBMISSIONS_DIR / "v53_best_raw_rbf_B_mean_working_sentinel99.csv",
        "v54_sentinel150": SUBMISSIONS_DIR / "v54_best_raw_rbf_sentinel_150.csv",
        "v54_sentinel999": SUBMISSIONS_DIR / "v54_best_raw_rbf_sentinel_999.csv",
    }
    loaded = {k: pd.read_csv(p) for k, p in candidates.items() if p.exists()}
    rows = []
    keys = list(loaded)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            merged = loaded[a].merge(loaded[b], on=ID_COL, suffixes=(f"_{a}", f"_{b}"))
            pa = merged[f"{TARGET}_{a}"]
            pb = merged[f"{TARGET}_{b}"]
            diff = (pa - pb).abs()
            rows.append(
                {
                    "left": a,
                    "right": b,
                    "mean_abs_diff": float(diff.mean()),
                    "max_abs_diff": float(diff.max()),
                    "num_different_rows": int((diff > 0).sum()),
                    "endpoint_changed_rows": int(((pa.isin([0, 1])) != (pb.isin([0, 1]))).sum()),
                    "notes": "QA only; not used for model selection.",
                }
            )
    return pd.DataFrame(rows)


def run_v60_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_summary, base_folds, base_oof = baseline_oof(train_df)
    base_summary.to_csv(REPORTS_DIR / "v60_baseline_oof_summary.csv", index=False)
    base_oof.to_csv(REPORTS_DIR / "oof_predictions_v60_baseline.csv", index=False)

    sentinel99_oof = base_oof[base_oof["candidate"].eq("sentinel99")].copy()
    df = analysis_frame(train_df, sentinel99_oof)
    miss_summary, miss_residual = missing_pattern_reports(df)
    miss_summary.to_csv(REPORTS_DIR / "v60_missing_pattern_summary.csv", index=False)
    miss_residual.to_csv(REPORTS_DIR / "v60_missing_pattern_residual.csv", index=False)

    miss_feature = missing_feature_test(train_df, miss_residual)
    miss_feature.to_csv(REPORTS_DIR / "v60_missing_pattern_feature_test.csv", index=False)

    grid_metrics, class_counts, decile_err = y100_grid_diagnostic(df)
    pd.concat([grid_metrics, decile_err], ignore_index=True).to_csv(REPORTS_DIR / "v60_y100_grid_diagnostic.csv", index=False)
    class_counts.to_csv(REPORTS_DIR / "v60_y100_class_counts.csv", index=False)
    ordinal_baseline(train_df).to_csv(REPORTS_DIR / "v60_ordinal_baseline_result.csv", index=False)

    calibration_diagnostic(df).to_csv(REPORTS_DIR / "v60_calibration_diagnostic.csv", index=False)
    score_approx, score_rules = score_form_approximation(df)
    score_approx.to_csv(REPORTS_DIR / "v60_score_form_approximation.csv", index=False)
    score_rules.to_csv(REPORTS_DIR / "v60_score_rule_candidates.csv", index=False)

    candidate = candidate_feature_test(train_df, miss_residual)
    candidate.to_csv(REPORTS_DIR / "v60_candidate_feature_test.csv", index=False)

    # Submission only if missing-pattern feature beats sentinel99 by >= 0.0005.
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    best = candidate.iloc[0]
    if best["mean_mae"] <= BASELINE_CV - 0.0005:
        # Rebuild only supported feature by name.
        model = make_missing_feature_pipeline(best["candidate"])
        test_df = pd.read_csv(TEST_PATH)
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
        pred = clip_round_2(model.predict(test_df))
        sub = sample.copy()
        sub[TARGET] = pred
        path = SUBMISSIONS_DIR / f"v60_best_raw_rbf_{best['candidate']}.csv"
        sub.to_csv(path, index=False)
        paths.append(path)

    submission_qa().to_csv(REPORTS_DIR / "v60_submission_disagreement_QA.csv", index=False)

    print("\n=== V6.0 baseline ===")
    print(base_summary.round(6).to_string(index=False))
    print("\n=== V6.0 missing pattern residual top ===")
    print(miss_residual.head(10).round(6).to_string(index=False))
    print("\n=== V6.0 missing feature test ===")
    print(miss_feature.round(6).to_string(index=False))
    print("\n=== V6.0 candidate feature test ===")
    print(candidate.round(6).to_string(index=False))
    if paths:
        for path in paths:
            print(f"Saved submission: {path}")
    else:
        print("No v6.0 submission: no feature improved baseline by >= 0.0005.")


if __name__ == "__main__":
    run_v60_experiments()
