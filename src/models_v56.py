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
from src.models_v54 import V54FeatureEngineer, apply_grid_postprocess, feature_columns
from src.postprocess import clip_round_2


EXP_ID = f"v56_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_CV = 0.13488666666666665
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


def base_config(sentinel):
    return {"mean_working_mode": "sentinel", "sentinel_value": float(sentinel)}


def make_preprocessor(config, extra_numeric=None):
    numeric, ohe = feature_columns(config)
    if extra_numeric:
        numeric = numeric + extra_numeric
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
    config = base_config(sentinel)
    return Pipeline(
        steps=[
            ("features", V54FeatureEngineer(**config)),
            ("preprocess", make_preprocessor(config)),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def evaluate_oof(train_df, name, pipeline):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_pred = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(SPLITTER.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_pred[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        pred = clip_round_2(raw_pred[va_idx])
        fold_rows.append(
            {
                "candidate": name,
                "fold": fold,
                "mae": mean_absolute_error(y[va_idx], pred),
            }
        )
    pred = clip_round_2(raw_pred)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
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
            "candidate": name,
            "fold": folds,
            "raw_oof_pred": raw_pred,
            "oof_pred": pred,
        }
    )
    return summary, pd.DataFrame(fold_rows), oof


def build_analysis_frame(train_df, oof):
    df = train_df.merge(oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left").copy()
    df["pred100"] = df["oof_pred"] * 100
    df["y100"] = df[TARGET] * 100
    df["residual100"] = df["y100"] - df["pred100"]
    df["abs_residual100"] = df["residual100"].abs()
    df["bmi"] = df["weight"] / np.square(df["height"] / 100)
    df["pulse_pressure"] = df["systolic_blood_pressure"] - df["diastolic_blood_pressure"]
    df["map"] = df["diastolic_blood_pressure"] + df["pulse_pressure"] / 3
    df["cholesterol_glucose_product"] = df["cholesterol"] * df["glucose"]
    df["glucose_cholesterol_ratio"] = df["glucose"] / df["cholesterol"].replace(0, np.nan)
    df["mean_working_missing"] = df["mean_working"].isna().astype("int8")
    df["mean_working_group"] = V54FeatureEngineer._mean_working_group(df["mean_working"])
    df["age_bin"] = pd.cut(df["age"], bins=[-np.inf, 40, 55, 70, 85, np.inf], labels=["<=40", "41_55", "56_70", "71_85", "86+"])
    df["BMI_bin"] = pd.cut(df["bmi"], bins=[-np.inf, 18.5, 23, 27.5, 32, np.inf], labels=["low", "normal", "high", "obese", "very_high"])
    df["glucose_bin"] = pd.qcut(df["glucose"], q=5, labels=["g0", "g1", "g2", "g3", "g4"], duplicates="drop")
    df["cholesterol_bin"] = pd.qcut(df["cholesterol"], q=5, labels=["c0", "c1", "c2", "c3", "c4"], duplicates="drop")
    df["BP_bin"] = pd.cut(df["systolic_blood_pressure"], bins=[-np.inf, 120, 140, 160, 180, np.inf], labels=["bp0", "bp1", "bp2", "bp3", "bp4"])
    df["bone_density_bin"] = pd.qcut(df["bone_density"], q=5, labels=["b0", "b1", "b2", "b3", "b4"], duplicates="drop")
    return df


def baseline_oof(train_df):
    rows, folds, oofs = [], [], []
    for sentinel in [99, 150]:
        name = f"sentinel{sentinel}"
        summary, fold_df, oof = evaluate_oof(train_df, name, make_base_pipeline(sentinel))
        rows.append(summary)
        folds.append(fold_df)
        oofs.append(oof)
        print(f"baseline {name}: {summary['mean_mae']:.6f}")
    summary_df = pd.DataFrame(rows).sort_values("mean_mae")
    fold_df = pd.concat(folds, ignore_index=True)
    oof_df = pd.concat(oofs, ignore_index=True)
    return summary_df, fold_df, oof_df


def residual_tree_rules(analysis_by_candidate):
    feature_cols = [
        "age",
        "height",
        "weight",
        "cholesterol",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "glucose",
        "bone_density",
        "bmi",
        "pulse_pressure",
        "map",
        "cholesterol_glucose_product",
        "glucose_cholesterol_ratio",
        "mean_working_missing",
    ]
    cat_cols = ["gender", "activity", "smoke_status", "medical_history", "family_medical_history", "sleep_pattern", "edu_level", "mean_working_group"]
    rows = []
    for candidate, df in analysis_by_candidate.items():
        X = df[feature_cols + cat_cols].copy()
        y = df["residual100"].to_numpy()
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_cols),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ],
            remainder="drop",
        )
        for depth in [2, 3, 4]:
            for leaf in [50, 80, 100]:
                pipe = Pipeline([("pre", pre), ("tree", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE))])
                pipe.fit(X, y)
                Xt = pipe.named_steps["pre"].transform(X)
                tree = pipe.named_steps["tree"]
                leaf_id = tree.apply(Xt)
                feature_names = pipe.named_steps["pre"].get_feature_names_out()
                rules = _tree_rules(tree, feature_names)
                for node_id, rule in rules.items():
                    mask = leaf_id == node_id
                    count = int(mask.sum())
                    if count < 30:
                        continue
                    sub = df.loc[mask]
                    rows.append(
                        {
                            "exp_id": EXP_ID,
                            "candidate": candidate,
                            "max_depth": depth,
                            "min_samples_leaf": leaf,
                            "rule": rule,
                            "count": count,
                            "residual_mean": float(sub["residual100"].mean()),
                            "residual_median": float(sub["residual100"].median()),
                            "residual_mae": float(sub["abs_residual100"].mean()),
                            "y_mean": float(sub[TARGET].mean()),
                            "pred_mean": float(sub["oof_pred"].mean()),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["candidate", "max_depth", "min_samples_leaf", "residual_mae"], ascending=[True, True, True, False])


def _tree_rules(tree, feature_names):
    tree_ = tree.tree_
    rules = {}

    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = " AND ".join(path) if path else "ALL"
            return
        name = feature_names[tree_.feature[node]]
        threshold = tree_.threshold[node]
        recurse(tree_.children_left[node], path + [f"{name} <= {threshold:.4f}"])
        recurse(tree_.children_right[node], path + [f"{name} > {threshold:.4f}"])

    recurse(0, [])
    return rules


def two_way_tables(df):
    pairs = [
        ("mean_working_group", "sleep_pattern"),
        ("mean_working_group", "activity"),
        ("mean_working_group", "smoke_status"),
        ("mean_working_group", "medical_history"),
        ("mean_working_group", "family_medical_history"),
        ("mean_working_group", "edu_level"),
        ("sleep_pattern", "activity"),
        ("smoke_status", "medical_history"),
        ("medical_history", "family_medical_history"),
        ("glucose_bin", "cholesterol_bin"),
        ("BMI_bin", "glucose_bin"),
        ("age_bin", "medical_history"),
        ("bone_density_bin", "gender"),
    ]
    rows = []
    for a, b in pairs:
        tmp = df.copy()
        tmp[a] = tmp[a].astype("object").fillna("Unknown")
        tmp[b] = tmp[b].astype("object").fillna("Unknown")
        grouped = tmp.groupby([a, b], dropna=False).agg(
            count=(TARGET, "size"),
            y_mean=(TARGET, "mean"),
            pred_mean=("oof_pred", "mean"),
            residual_mean=("residual100", "mean"),
            residual_median=("residual100", "median"),
            residual_mae=("abs_residual100", "mean"),
        ).reset_index()
        grouped = grouped.rename(columns={a: "left_value", b: "right_value"})
        grouped.insert(0, "left_feature", a)
        grouped.insert(1, "right_feature", b)
        grouped["notes"] = np.where(grouped["count"] < 30, "small_n", "")
        rows.append(grouped)
    table = pd.concat(rows, ignore_index=True)
    candidates = table[(table["count"] >= 30) & (table["residual_mean"].abs() >= 5)].copy()
    candidates = candidates.sort_values("residual_mean", key=lambda s: s.abs(), ascending=False)
    candidates.insert(0, "exp_id", EXP_ID)
    table.insert(0, "exp_id", EXP_ID)
    return table, candidates


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, sentinel=99, interactions=None):
        self.sentinel = sentinel
        self.interactions = interactions or []
        self.base = V54FeatureEngineer(mean_working_mode="sentinel", sentinel_value=float(sentinel))

    def fit(self, X, y=None):
        self.base.fit(X, y)
        return self

    def transform(self, X):
        base_df = self.base.transform(X)
        raw = X.copy()
        helper = build_helper_features(raw)
        for idx, spec in enumerate(self.interactions):
            a, av, b, bv = spec
            base_df[f"rule_feature_{idx}"] = ((helper[a].astype(str) == str(av)) & (helper[b].astype(str) == str(bv))).astype("int8")
        return base_df


def build_helper_features(df):
    out = df.copy()
    out["mean_working_group"] = V54FeatureEngineer._mean_working_group(out["mean_working"])
    out["bmi"] = out["weight"] / np.square(out["height"] / 100)
    out["age_bin"] = pd.cut(out["age"], bins=[-np.inf, 40, 55, 70, 85, np.inf], labels=["<=40", "41_55", "56_70", "71_85", "86+"]).astype("object")
    out["BMI_bin"] = pd.cut(out["bmi"], bins=[-np.inf, 18.5, 23, 27.5, 32, np.inf], labels=["low", "normal", "high", "obese", "very_high"]).astype("object")
    out["glucose_bin"] = pd.cut(out["glucose"], bins=[-np.inf, 120, 135, 150, 165, np.inf], labels=["g0", "g1", "g2", "g3", "g4"]).astype("object")
    out["cholesterol_bin"] = pd.cut(out["cholesterol"], bins=[-np.inf, 220, 260, 300, 340, np.inf], labels=["c0", "c1", "c2", "c3", "c4"]).astype("object")
    out["bone_density_bin"] = pd.cut(out["bone_density"], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=["b0", "b1", "b2", "b3", "b4"]).astype("object")
    return out


def make_interaction_pipeline(sentinel, interactions):
    config = base_config(sentinel)
    extra = [f"rule_feature_{i}" for i in range(len(interactions))]
    return Pipeline(
        [
            ("features", InteractionFeatureEngineer(sentinel=sentinel, interactions=interactions)),
            ("preprocess", make_preprocessor(config, extra_numeric=extra)),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def select_interactions(candidates):
    specs = []
    for _, row in candidates.iterrows():
        lf, rf = row["left_feature"], row["right_feature"]
        if row["count"] < 30:
            continue
        specs.append((lf, row["left_value"], rf, row["right_value"]))
        if len(specs) >= 5:
            break
    return specs


def candidate_interaction_test(train_df, specs):
    rows = []
    oofs = []
    for idx, spec in enumerate(specs):
        name = f"rule_{idx}_{spec[0]}={spec[1]}__{spec[2]}={spec[3]}"
        summary, _, oof = evaluate_oof(train_df, name, make_interaction_pipeline(99, [spec]))
        summary.update({"rule": name, "num_rule_features": 1, "notes": "single residual-derived binary rule feature"})
        rows.append(summary)
        oofs.append(oof)
        print(f"interaction {idx}: {summary['mean_mae']:.6f} {name}")
    if specs:
        summary, _, oof = evaluate_oof(train_df, "top_rules_all", make_interaction_pipeline(99, specs))
        summary.update({"rule": "top_rules_all", "num_rule_features": len(specs), "notes": "top residual-derived binary rule features together"})
        rows.append(summary)
        oofs.append(oof)
    return pd.DataFrame(rows).sort_values("mean_mae"), pd.concat(oofs, ignore_index=True) if oofs else pd.DataFrame()


def residual_correction_simulation(df, hidden_candidates):
    rows = []
    y = df[TARGET].to_numpy()
    base_pred = df["oof_pred"].to_numpy()
    for _, row in hidden_candidates.head(10).iterrows():
        mask = (
            (df[row["left_feature"]].astype("object").fillna("Unknown").astype(str) == str(row["left_value"]))
            & (df[row["right_feature"]].astype("object").fillna("Unknown").astype(str) == str(row["right_value"]))
        )
        if mask.sum() < 50:
            continue
        residual = row["residual_mean"] / 100.0
        for frac in [0.1, 0.2, 0.3]:
            pred = base_pred.copy()
            pred[mask.to_numpy()] = apply_grid_postprocess(pred[mask.to_numpy()] + frac * residual, "round2")
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "left_feature": row["left_feature"],
                    "left_value": row["left_value"],
                    "right_feature": row["right_feature"],
                    "right_value": row["right_value"],
                    "count": int(mask.sum()),
                    "residual_mean": row["residual_mean"],
                    "correction_fraction": frac,
                    "mae": float(mean_absolute_error(y, pred)),
                    "notes": "OOF-only diagnostic; not used for submission",
                }
            )
    return pd.DataFrame(rows).sort_values("mae") if rows else pd.DataFrame()


def maybe_save_submission(train_df, test_df, sample_submission, interaction_df, specs):
    if interaction_df.empty:
        return []
    best = interaction_df.iloc[0]
    if best["mean_mae"] > BASELINE_CV - 0.0005:
        return []
    if best["candidate"] == "top_rules_all":
        selected = specs
        suffix = "top_rules_all"
    else:
        idx = int(best["candidate"].split("_")[1])
        selected = [specs[idx]]
        suffix = f"rule_{idx}"
    model = make_interaction_pipeline(99, selected)
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    pred = clip_round_2(model.predict(test_df))
    sub = sample_submission.copy()
    sub[TARGET] = pred
    path = SUBMISSIONS_DIR / f"v56_best_raw_rbf_residual_rule_{suffix}.csv"
    sub.to_csv(path, index=False)
    return [path]


def run_v56_experiments():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, fold_df, oof_df = baseline_oof(train_df)
    oof_df.to_csv(REPORTS_DIR / "oof_predictions_v56_baseline.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "v56_baseline_oof_summary.csv", index=False)
    fold_df.to_csv(REPORTS_DIR / "v56_baseline_fold_mae.csv", index=False)

    analysis_by_candidate = {}
    for candidate in ["sentinel99", "sentinel150"]:
        sub_oof = oof_df[oof_df["candidate"].eq(candidate)]
        analysis_by_candidate[candidate] = build_analysis_frame(train_df, sub_oof)

    tree_df = residual_tree_rules(analysis_by_candidate)
    tree_df.to_csv(REPORTS_DIR / "v56_residual_tree_rules.csv", index=False)

    main_df = analysis_by_candidate["sentinel99"]
    two_way, hidden = two_way_tables(main_df)
    two_way.to_csv(REPORTS_DIR / "v56_two_way_residual_table.csv", index=False)
    hidden.to_csv(REPORTS_DIR / "v56_hidden_score_item_candidates.csv", index=False)

    specs = select_interactions(hidden)
    interaction_df, interaction_oof = candidate_interaction_test(train_df, specs)
    interaction_df.to_csv(REPORTS_DIR / "v56_candidate_interaction_test.csv", index=False)
    if not interaction_oof.empty:
        interaction_oof.to_csv(REPORTS_DIR / "oof_predictions_v56_interactions.csv", index=False)

    correction_df = residual_correction_simulation(main_df, hidden)
    correction_df.to_csv(REPORTS_DIR / "v56_residual_correction_simulation.csv", index=False)

    paths = maybe_save_submission(train_df, test_df, sample_submission, interaction_df, specs)

    print("\n=== V5.6 baseline ===")
    print(summary_df.round(6).to_string(index=False))
    print("\n=== V5.6 hidden score candidates ===")
    print(hidden.head(15).round(6).to_string(index=False))
    print("\n=== V5.6 candidate interaction test ===")
    print(interaction_df.round(6).to_string(index=False) if not interaction_df.empty else "No interaction candidates.")
    print("\n=== V5.6 correction simulation top ===")
    print(correction_df.head(10).round(6).to_string(index=False) if not correction_df.empty else "No correction candidates.")
    if paths:
        for path in paths:
            print(f"Saved submission: {path}")
    else:
        print("No v5.6 submission: no residual-rule feature improved baseline by >= 0.0005.")


if __name__ == "__main__":
    run_v56_experiments()
