from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config import ID_COL, REPORTS_DIR, TARGET, TRAIN_PATH
from src.models_v5 import DenseTransformer
from src.models_v54 import apply_grid_postprocess
from src.models_v81 import get_rbf_oof, mean_working_group
from src.models_v8 import feature_preprocessor


EXP_ID = f"v83_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_SPLITS = 10
RANDOM_STATE = 42

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
    "glucose_cholesterol_ratio",
    "cholesterol_glucose_product",
    "gender_code",
]
BASE_CATEGORICAL = [
    "activity_cat",
    "sleep_pattern_cat",
    "edu_level_cat",
    "smoke_status_cat",
    "medical_history_cat",
    "family_medical_history_cat",
]
RULE_FEATURES = [
    "rule_mw11_med_missing",
    "rule_mw10_med_heart",
    "rule_mw11_family_missing",
    "rule_mw7_sleep_oversleeping",
    "rule_mw11_current_smoker",
    "rule_mw9_med_diabetes",
    "rule_mw12plus_med_missing",
    "rule_mw10_family_diabetes",
    "mw_low_6_or_less",
    "mw_high_11",
    "mw_high_12plus",
]


class V83FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_rules=False):
        self.add_rules = add_rules

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        for col in [
            "gender",
            "activity",
            "smoke_status",
            "medical_history",
            "family_medical_history",
            "sleep_pattern",
            "edu_level",
        ]:
            X_df[f"{col}_cat"] = X_df[col].astype("object").fillna("Unknown")
        raw_mw = X_df["mean_working"]
        X_df["mean_working_group"] = mean_working_group(raw_mw)
        X_df["mean_working"] = raw_mw.fillna(99.0)
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]
        X_df["gender_code"] = X_df["gender_cat"].map({"F": 0, "M": 1}).astype(float)

        if self.add_rules:
            med = X_df["medical_history_cat"]
            fam = X_df["family_medical_history_cat"]
            sleep = X_df["sleep_pattern_cat"]
            smoke = X_df["smoke_status_cat"]
            mw = X_df["mean_working_group"]
            X_df["rule_mw11_med_missing"] = ((mw == "11") & (med == "Unknown")).astype("int8")
            X_df["rule_mw10_med_heart"] = ((mw == "10") & (med == "heart disease")).astype("int8")
            X_df["rule_mw11_family_missing"] = ((mw == "11") & (fam == "Unknown")).astype("int8")
            X_df["rule_mw7_sleep_oversleeping"] = ((mw == "7") & (sleep == "oversleeping")).astype("int8")
            X_df["rule_mw11_current_smoker"] = ((mw == "11") & (smoke == "current-smoker")).astype("int8")
            X_df["rule_mw9_med_diabetes"] = ((mw == "9") & (med == "diabetes")).astype("int8")
            X_df["rule_mw12plus_med_missing"] = ((mw == ">=12") & (med == "Unknown")).astype("int8")
            X_df["rule_mw10_family_diabetes"] = ((mw == "10") & (fam == "diabetes")).astype("int8")
            X_df["mw_low_6_or_less"] = (mw == "<=6").astype("int8")
            X_df["mw_high_11"] = (mw == "11").astype("int8")
            X_df["mw_high_12plus"] = (mw == ">=12").astype("int8")
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])


class Y100Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y * 100.0)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) / 100.0


def make_preprocessor(add_rules=False):
    numeric = BASE_NUMERIC.copy()
    if add_rules:
        numeric += RULE_FEATURES
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
                BASE_CATEGORICAL,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_pipeline(estimator, add_rules=False):
    return Pipeline(
        steps=[
            ("features", V83FeatureEngineer(add_rules=add_rules)),
            ("preprocess", make_preprocessor(add_rules=add_rules)),
            ("dense", DenseTransformer()),
            ("model", Y100Regressor(estimator)),
        ]
    )


def candidates():
    return [
        (
            "ridge_y100_base",
            make_pipeline(Ridge(alpha=1.0), add_rules=False),
            "linear",
            False,
        ),
        (
            "ridge_y100_rule_features",
            make_pipeline(Ridge(alpha=1.0), add_rules=True),
            "linear_rules",
            True,
        ),
        (
            "decision_tree_depth4",
            make_pipeline(DecisionTreeRegressor(max_depth=4, min_samples_leaf=40, random_state=RANDOM_STATE), add_rules=False),
            "tree",
            False,
        ),
        (
            "decision_tree_depth6",
            make_pipeline(DecisionTreeRegressor(max_depth=6, min_samples_leaf=25, random_state=RANDOM_STATE), add_rules=False),
            "tree",
            False,
        ),
        (
            "decision_tree_depth8",
            make_pipeline(DecisionTreeRegressor(max_depth=8, min_samples_leaf=15, random_state=RANDOM_STATE), add_rules=False),
            "tree",
            False,
        ),
        (
            "decision_tree_depth6_rules",
            make_pipeline(DecisionTreeRegressor(max_depth=6, min_samples_leaf=25, random_state=RANDOM_STATE), add_rules=True),
            "tree_rules",
            True,
        ),
        (
            "extra_trees_leaf5",
            make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=600,
                    min_samples_leaf=5,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                add_rules=False,
            ),
            "ensemble_tree",
            False,
        ),
        (
            "extra_trees_leaf5_rules",
            make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=600,
                    min_samples_leaf=5,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                add_rules=True,
            ),
            "ensemble_tree_rules",
            True,
        ),
        (
            "random_forest_leaf5",
            make_pipeline(
                RandomForestRegressor(
                    n_estimators=600,
                    min_samples_leaf=5,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                add_rules=False,
            ),
            "ensemble_tree",
            False,
        ),
        (
            "hist_gradient_boosting",
            make_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
                add_rules=False,
            ),
            "boosting",
            False,
        ),
        (
            "hist_gradient_boosting_rules",
            make_pipeline(
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=15,
                    l2_regularization=0.05,
                    random_state=RANDOM_STATE,
                ),
                add_rules=True,
            ),
            "boosting_rules",
            True,
        ),
    ]


def evaluate_candidate(train_df, name, pipeline):
    splitter = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df))
    folds = np.zeros(len(train_df), dtype=int)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
    pred = apply_grid_postprocess(raw_oof, "round2")
    for fold in range(1, 11):
        mask = folds == fold
        rows.append({"exp_id": EXP_ID, "candidate": name, "fold": fold, "mae": mean_absolute_error(y[mask], pred[mask])})
    fold_df = pd.DataFrame(rows)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=1)),
        "raw_mae": float(mean_absolute_error(y, raw_oof)),
        "round2_mae": float(mean_absolute_error(y, pred)),
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


def fit_full_importance_and_rules(train_df):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    pipe = make_pipeline(
        ExtraTreesRegressor(
            n_estimators=800,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        add_rules=True,
    )
    pipe.fit(X, y)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "exp_id": EXP_ID,
            "feature": feature_names,
            "importance": pipe.named_steps["model"].estimator_.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    tree_pipe = make_pipeline(
        DecisionTreeRegressor(max_depth=6, min_samples_leaf=25, random_state=RANDOM_STATE),
        add_rules=True,
    )
    tree_pipe.fit(X, y)
    tree_feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
    rules = export_text(
        tree_pipe.named_steps["model"].estimator_,
        feature_names=list(tree_feature_names),
        max_depth=6,
    )
    return importances, rules


def surrogate_to_rbf(train_df):
    rbf = get_rbf_oof(train_df)
    X = train_df.drop(columns=[TARGET])
    y_rbf = rbf["oof_pred"].to_numpy()
    prep = feature_preprocessor(dense=True)
    Xf = prep.fit_transform(X)
    names = prep.named_steps["preprocess"].get_feature_names_out()
    rows = []
    rules = {}
    for depth in [4, 6, 8]:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=20, random_state=RANDOM_STATE)
        tree.fit(Xf, y_rbf)
        pred = tree.predict(Xf)
        rows.append(
            {
                "exp_id": EXP_ID,
                "surrogate": f"rbf_surrogate_tree_depth{depth}",
                "mae_to_rbf": float(mean_absolute_error(y_rbf, pred)),
                "corr_to_rbf": float(np.corrcoef(y_rbf, pred)[0, 1]),
                "r2_to_rbf": float(r2_score(y_rbf, pred)),
            }
        )
        rules[depth] = export_text(tree, feature_names=list(names), max_depth=depth)
    extra = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    extra.fit(Xf, y_rbf)
    pred = extra.predict(Xf)
    rows.append(
        {
            "exp_id": EXP_ID,
            "surrogate": "rbf_surrogate_extra_trees_leaf3",
            "mae_to_rbf": float(mean_absolute_error(y_rbf, pred)),
            "corr_to_rbf": float(np.corrcoef(y_rbf, pred)[0, 1]),
            "r2_to_rbf": float(r2_score(y_rbf, pred)),
        }
    )
    imp = pd.DataFrame({"exp_id": EXP_ID, "feature": names, "importance": extra.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return pd.DataFrame(rows), imp, rules


def compare_rule_features(summary_df):
    rows = []
    pairs = [
        ("ridge_y100_base", "ridge_y100_rule_features"),
        ("decision_tree_depth6", "decision_tree_depth6_rules"),
        ("extra_trees_leaf5", "extra_trees_leaf5_rules"),
        ("hist_gradient_boosting", "hist_gradient_boosting_rules"),
    ]
    lookup = summary_df.set_index("candidate")
    for base, rules in pairs:
        if base in lookup.index and rules in lookup.index:
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "base_candidate": base,
                    "rule_candidate": rules,
                    "base_mae": lookup.loc[base, "mean_mae"],
                    "rule_mae": lookup.loc[rules, "mean_mae"],
                    "improvement_from_rules": lookup.loc[base, "mean_mae"] - lookup.loc[rules, "mean_mae"],
                }
            )
    return pd.DataFrame(rows)


def build_synthesis(summary, rule_compare, importances, surrogate_fidelity, surrogate_importance):
    best = summary.sort_values("mean_mae").iloc[0]
    best_rule = rule_compare.sort_values("improvement_from_rules", ascending=False).iloc[0]
    top_importance = importances.head(20)
    top_rule_features = importances[importances["feature"].str.contains("rule_|mw_", regex=True)].head(20)
    top_surrogate = surrogate_fidelity.sort_values("mae_to_rbf").iloc[0]
    top_surrogate_importance = surrogate_importance.head(20)
    return f"""# v8.3 final suspicion check: tree/boost/rule models

## 1. y100 직접 모델 비교
```text
{summary[['candidate', 'family', 'add_rules', 'mean_mae', 'pred_std', 'pred_min', 'pred_max']].sort_values('mean_mae').to_string(index=False)}
```

가장 좋은 해석형 계열 후보는 `{best['candidate']}`이며 CV MAE는 {best['mean_mae']:.6f}입니다.
RBF의 0.13417에는 못 미치지만, tree/boost 계열이 Ridge보다 확실히 낫다면 단순 선형식보다 threshold/interaction 구조가 더 그럴듯합니다.

## 2. v8.2 rule 후보를 명시적으로 넣었을 때
```text
{rule_compare.to_string(index=False)}
```

가장 큰 rule feature 개선은 `{best_rule['rule_candidate']}`이며 개선폭은 {best_rule['improvement_from_rules']:.6f}입니다.
개선이 작거나 음수이면, v8.2 rule 후보는 해석 신호이지만 단순 binary feature 몇 개만으로 성능을 끌어올리는 구조는 아닙니다.

## 3. 실제 y100 ExtraTrees 중요도 상위
```text
{top_importance[['feature', 'importance']].to_string(index=False)}
```

명시 rule/tail feature 중요도:

```text
{top_rule_features[['feature', 'importance']].to_string(index=False)}
```

## 4. RBF surrogate
RBF OOF를 가장 잘 흉내낸 surrogate는 `{top_surrogate['surrogate']}`이며 MAE는 {top_surrogate['mae_to_rbf']:.6f}, correlation은 {top_surrogate['corr_to_rbf']:.6f}입니다.

```text
{surrogate_fidelity.to_string(index=False)}
```

RBF surrogate ExtraTrees 중요도 상위:

```text
{top_surrogate_importance[['feature', 'importance']].to_string(index=False)}
```

## 5. 최종 판단
- 0~100 점수식 의심은 유지됩니다.
- 단순 선형식은 Ridge/target transform 계열이 약해서 가능성이 낮습니다.
- tree/boost가 Ridge보다 낫고, RBF surrogate를 ensemble tree가 잘 흉내내면 구간/상호작용/비선형 score surface 가설이 강화됩니다.
- 하지만 명시 rule feature 몇 개만으로 큰 개선이 없다면, 사람이 한두 줄로 쓸 수 있는 단순 rule list가 아니라 많은 약한 threshold와 interaction이 합쳐진 복합 생성식일 가능성이 큽니다.
- 따라서 제출 모델에서 RBF가 암묵적으로 먹은 것은 숨은 점수식의 일부 흔적이며, 이를 완전히 손으로 복원하기는 어렵다는 쪽으로 의심을 내려놓는 것이 합리적입니다.
"""


def run_v83():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    summaries = []
    folds = []
    oofs = []
    for name, pipe, family, add_rules in candidates():
        summary, fold_df, oof = evaluate_candidate(train_df, name, pipe)
        summary["family"] = family
        summary["add_rules"] = add_rules
        summaries.append(summary)
        folds.append(fold_df.assign(family=family, add_rules=add_rules))
        oofs.append(oof.assign(family=family, add_rules=add_rules))
        print(f"{name}: {summary['mean_mae']:.6f}")

    summary_df = pd.DataFrame(summaries).sort_values("mean_mae")
    summary_df.to_csv(REPORTS_DIR / "v83_tree_boost_y100_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(folds, ignore_index=True).to_csv(REPORTS_DIR / "v83_tree_boost_y100_fold_results.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_csv(REPORTS_DIR / "v83_tree_boost_y100_oof.csv", index=False)

    rule_compare = compare_rule_features(summary_df)
    rule_compare.to_csv(REPORTS_DIR / "v83_rule_feature_validation.csv", index=False, encoding="utf-8-sig")

    importance, tree_rules = fit_full_importance_and_rules(train_df)
    importance.to_csv(REPORTS_DIR / "v83_y100_extra_trees_rule_feature_importance.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "v83_y100_decision_tree_rules.txt").write_text(tree_rules, encoding="utf-8-sig")

    surrogate_fidelity, surrogate_importance, surrogate_rules = surrogate_to_rbf(train_df)
    surrogate_fidelity.to_csv(REPORTS_DIR / "v83_rbf_surrogate_fidelity.csv", index=False, encoding="utf-8-sig")
    surrogate_importance.to_csv(REPORTS_DIR / "v83_rbf_surrogate_feature_importance.csv", index=False, encoding="utf-8-sig")
    for depth, text in surrogate_rules.items():
        (REPORTS_DIR / f"v83_rbf_surrogate_tree_depth{depth}_rules.txt").write_text(text, encoding="utf-8-sig")

    synthesis = build_synthesis(summary_df, rule_compare, importance, surrogate_fidelity, surrogate_importance)
    (REPORTS_DIR / "v83_final_suspicion_check_synthesis.md").write_text(synthesis, encoding="utf-8-sig")

    print("\n=== v8.3 model comparison ===")
    print(summary_df[["candidate", "family", "add_rules", "mean_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== v8.3 rule feature validation ===")
    print(rule_compare.round(6).to_string(index=False))
    print("\n=== v8.3 surrogate fidelity ===")
    print(surrogate_fidelity.round(6).to_string(index=False))
    return summary_df


if __name__ == "__main__":
    run_v83()
