from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import (
    ID_COL,
    REPORTS_DIR,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.models_v54 import V53_SENTINEL99_CV, apply_grid_postprocess, rbf_estimator


EXP_ID = f"v532_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASELINE_LB = 0.13023
N_SPLITS = 10
RANDOM_STATE = 42

RAW_NUMERIC = [
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
DERIVED_FEATURES = {
    "bmi": ["bmi"],
    "bp": ["pulse_pressure", "map"],
    "ratio": ["glucose_cholesterol_ratio"],
    "product": ["cholesterol_glucose_product"],
}
FEATURE_SETS = {
    "FS0_baseline_S2_all": ["bmi", "bp", "ratio", "product"],
    "FS1_raw_plus_BMI_only": ["bmi"],
    "FS2_raw_plus_BMI_BP": ["bmi", "bp"],
    "FS3_raw_plus_BMI_metabolic_product": ["bmi", "product"],
    "FS4_raw_plus_BMI_metabolic_ratio": ["bmi", "ratio"],
    "FS5_raw_only_no_derived": [],
    "FS6_no_ratio_keep_product_BP": ["bmi", "bp", "product"],
    "FS7_no_product_keep_ratio_BP": ["bmi", "bp", "ratio"],
    "FS8_no_BP_keep_BMI_metabolic": ["bmi", "ratio", "product"],
}
ENCODINGS = {
    "ENC0_v53_current": "current",
    "ENC1_all_onehot": "all_onehot",
    "ENC2_clinical_order": "clinical_order",
    "ENC3_sleep_revised": "sleep_revised",
    "ENC4_no_ordinal_except_binary": "no_ordinal_except_binary",
    "ENC5_missing_explicit_for_edu_only": "missing_explicit_for_edu_only",
}


class V532FeatureEngineer(BaseEstimator, TransformerMixin):
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
        X_df["edu_level_explicit_cat"] = X_df["edu_level"].astype("object").where(
            X_df["edu_level"].notna(), "__MISSING__"
        )

        X_df["mean_working"] = X_df["mean_working"].fillna(99.0)
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)
        X_df["pulse_pressure"] = X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]

        X_df["gender_code"] = X_df["gender_cat"].map({"F": 0, "M": 1}).astype(float)
        X_df["activity_current_code"] = X_df["activity_cat"].map(
            {"light": 0, "moderate": 1, "intense": 2}
        ).astype(float)
        X_df["sleep_current_code"] = X_df["sleep_pattern_cat"].map(
            {"sleep difficulty": 0, "normal": 1, "oversleeping": 2}
        ).astype(float)
        X_df["sleep_risk_code"] = X_df["sleep_pattern_cat"].map(
            {"oversleeping": 0, "normal": 1, "sleep difficulty": 2}
        ).astype(float)
        X_df["edu_code"] = X_df["edu_level_cat"].map(
            {
                "Unknown": 0,
                "high school diploma": 1,
                "bachelors degree": 2,
                "graduate degree": 3,
            }
        ).astype(float)
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])


def feature_numeric_columns(feature_set_name, encoding_name):
    groups = FEATURE_SETS[feature_set_name]
    numeric = RAW_NUMERIC.copy()
    for group in groups:
        numeric.extend(DERIVED_FEATURES[group])

    mode = ENCODINGS[encoding_name]
    if mode in {"current", "clinical_order", "sleep_revised"}:
        numeric.extend(["gender_code", "activity_current_code", "edu_code"])
        numeric.append("sleep_risk_code" if mode in {"clinical_order", "sleep_revised"} else "sleep_current_code")
    elif mode in {"no_ordinal_except_binary", "missing_explicit_for_edu_only"}:
        numeric.append("gender_code")
    return numeric


def feature_ohe_columns(encoding_name):
    base = ["smoke_status_cat", "medical_history_cat", "family_medical_history_cat"]
    mode = ENCODINGS[encoding_name]
    if mode == "all_onehot":
        return ["gender_cat", "activity_cat", "sleep_pattern_cat", "edu_level_cat"] + base
    if mode == "no_ordinal_except_binary":
        return ["activity_cat", "sleep_pattern_cat", "edu_level_cat"] + base
    if mode == "missing_explicit_for_edu_only":
        return ["activity_cat", "sleep_pattern_cat", "edu_level_explicit_cat"] + base
    return base


def make_pipeline(feature_set_name, encoding_name):
    numeric_cols = feature_numeric_columns(feature_set_name, encoding_name)
    ohe_cols = feature_ohe_columns(encoding_name)
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
            ("features", V532FeatureEngineer()),
            ("preprocess", preprocessor),
            ("dense", DenseTransformer()),
            ("model", TargetModeRegressor(rbf_estimator(), target_mode="raw")),
        ]
    )


def binwise_mae(y, pred):
    bins = {
        "mae_y_le_0_1": y <= 0.1,
        "mae_y_0_1_0_3": (y > 0.1) & (y <= 0.3),
        "mae_y_0_3_0_7": (y > 0.3) & (y <= 0.7),
        "mae_y_0_7_0_9": (y > 0.7) & (y < 0.9),
        "mae_y_ge_0_9": y >= 0.9,
    }
    rows = {}
    for name, mask in bins.items():
        rows[name] = float(mean_absolute_error(y[mask], pred[mask])) if np.any(mask) else np.nan
    return rows


def summarize_oof(y, raw_pred, round_pred):
    clipped = apply_grid_postprocess(raw_pred, "clip")
    return {
        "clip_mae": float(mean_absolute_error(y, clipped)),
        "clip_round2_mae": float(mean_absolute_error(y, round_pred)),
        "pred_mean": float(np.mean(round_pred)),
        "pred_std": float(np.std(round_pred, ddof=1)),
        "pred_min": float(np.min(round_pred)),
        "pred_max": float(np.max(round_pred)),
        "pred_endpoint_0_count": int(np.sum(round_pred == 0)),
        "pred_endpoint_1_count": int(np.sum(round_pred == 1)),
        **binwise_mae(y, round_pred),
    }


def evaluate_cv(train_df, name, feature_set_name, encoding_name, test_df=None):
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    test_fold_raw = []
    base_pipeline = make_pipeline(feature_set_name, encoding_name)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(base_pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        if test_df is not None:
            test_fold_raw.append(model.predict(test_df))

    round_oof = apply_grid_postprocess(raw_oof, "round2")
    for fold in range(1, N_SPLITS + 1):
        mask = folds == fold
        fold_rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "feature_set": feature_set_name,
                "encoding": encoding_name,
                "fold": fold,
                "mae": float(mean_absolute_error(y[mask], round_oof[mask])),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "feature_set": feature_set_name,
        "encoding": encoding_name,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=1)),
        "baseline_cv": V53_SENTINEL99_CV,
        "improvement_vs_baseline": float(V53_SENTINEL99_CV - fold_df["mae"].mean()),
        **summarize_oof(y, raw_oof, round_oof),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": name,
            "feature_set": feature_set_name,
            "encoding": encoding_name,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": round_oof,
        }
    )
    test_round = None
    if test_df is not None:
        test_raw = np.mean(np.vstack(test_fold_raw), axis=0)
        test_round = apply_grid_postprocess(test_raw, "round2")
    return summary, fold_df, oof, test_round


def fit_full_train_predict(train_df, test_df, feature_set_name, encoding_name):
    model = make_pipeline(feature_set_name, encoding_name)
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    raw_pred = model.predict(test_df)
    return apply_grid_postprocess(raw_pred, "round2")


def submission_diff(left_name, left_pred, right_name, right_pred):
    left = np.asarray(left_pred)
    right = np.asarray(right_pred)
    diff = np.abs(left - right)
    return {
        "left": left_name,
        "right": right_name,
        "different_row_count": int(np.sum(left != right)),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "prediction_correlation": float(np.corrcoef(left, right)[0, 1]),
        "left_pred_mean": float(np.mean(left)),
        "left_pred_std": float(np.std(left, ddof=1)),
        "right_pred_mean": float(np.mean(right)),
        "right_pred_std": float(np.std(right, ddof=1)),
        "left_endpoint_0_count": int(np.sum(left == 0)),
        "left_endpoint_1_count": int(np.sum(left == 1)),
        "right_endpoint_0_count": int(np.sum(right == 0)),
        "right_endpoint_1_count": int(np.sum(right == 1)),
    }


def add_decision_columns(df):
    out = df.copy()
    out["candidate_level"] = np.select(
        [
            out["improvement_vs_baseline"] >= 0.001,
            out["improvement_vs_baseline"] >= 0.0005,
            out["improvement_vs_baseline"].between(0.0001, 0.0003, inclusive="both"),
            out["improvement_vs_baseline"] > 0,
        ],
        ["strong_submission_candidate", "submission_candidate", "micro_candidate", "tiny_micro_candidate"],
        default="hold",
    )
    out["std_warning"] = out["pred_std"] < 0.18
    out["endpoint_warning"] = (out["pred_endpoint_0_count"] < 3) | (out["pred_endpoint_1_count"] < 3)
    return out


def write_submission(sample_submission, pred, path):
    sub = sample_submission.copy()
    sub[TARGET] = pred
    sub.to_csv(path, index=False)


def run_v532():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    baseline_summary, baseline_folds, baseline_oof, _ = evaluate_cv(
        train_df,
        "FS0_baseline_S2_all__ENC0_v53_current",
        "FS0_baseline_S2_all",
        "ENC0_v53_current",
    )
    pd.DataFrame([baseline_summary]).to_csv(
        REPORTS_DIR / "v532_baseline_reproduce_cv.csv", index=False, encoding="utf-8-sig"
    )
    baseline_oof.to_csv(REPORTS_DIR / "v532_baseline_reproduce_oof.csv", index=False)

    ablation_rows = []
    ablation_folds = []
    ablation_oof_summaries = []
    for fs_name in FEATURE_SETS:
        name = f"{fs_name}__ENC0_v53_current"
        summary, folds, oof, _ = evaluate_cv(train_df, name, fs_name, "ENC0_v53_current")
        ablation_rows.append(summary)
        ablation_folds.append(folds)
        ablation_oof_summaries.append(summary)
        print(f"feature {fs_name}: {summary['mean_mae']:.6f}")

    ablation_cv = add_decision_columns(pd.DataFrame(ablation_rows).sort_values("mean_mae"))
    ablation_fold_df = pd.concat(ablation_folds, ignore_index=True)
    ablation_oof_summary = pd.DataFrame(ablation_oof_summaries).sort_values("mean_mae")
    ablation_cv.to_csv(REPORTS_DIR / "v532_feature_ablation_cv.csv", index=False, encoding="utf-8-sig")
    ablation_fold_df.to_csv(REPORTS_DIR / "v532_feature_ablation_fold_results.csv", index=False)
    ablation_oof_summary.to_csv(REPORTS_DIR / "v532_feature_ablation_oof_summary.csv", index=False)

    best_feature_set = ablation_cv.iloc[0]["feature_set"]
    encoding_rows = []
    encoding_folds = []
    encoding_oof_summaries = []
    baseline_oof_pred = baseline_oof["oof_pred"].to_numpy()
    baseline_full_pred = fit_full_train_predict(train_df, test_df, "FS0_baseline_S2_all", "ENC0_v53_current")
    for enc_name in ENCODINGS:
        name = f"{best_feature_set}__{enc_name}"
        summary, folds, oof, _ = evaluate_cv(train_df, name, best_feature_set, enc_name)
        summary["baseline_oof_correlation"] = float(np.corrcoef(baseline_oof_pred, oof["oof_pred"].to_numpy())[0, 1])
        candidate_full_pred = fit_full_train_predict(train_df, test_df, best_feature_set, enc_name)
        diff = submission_diff("baseline_sentinel99", baseline_full_pred, name, candidate_full_pred)
        summary["baseline_test_mean_abs_diff"] = diff["mean_abs_diff"]
        summary["baseline_test_max_abs_diff"] = diff["max_abs_diff"]
        summary["baseline_test_correlation"] = diff["prediction_correlation"]
        encoding_rows.append(summary)
        encoding_folds.append(folds)
        encoding_oof_summaries.append(summary)
        print(f"encoding {enc_name}: {summary['mean_mae']:.6f}")

    encoding_cv = add_decision_columns(pd.DataFrame(encoding_rows).sort_values("mean_mae"))
    encoding_fold_df = pd.concat(encoding_folds, ignore_index=True)
    encoding_oof_summary = pd.DataFrame(encoding_oof_summaries).sort_values("mean_mae")
    encoding_cv.to_csv(REPORTS_DIR / "v532_encoding_cv.csv", index=False, encoding="utf-8-sig")
    encoding_fold_df.to_csv(REPORTS_DIR / "v532_encoding_fold_results.csv", index=False)
    encoding_oof_summary.to_csv(REPORTS_DIR / "v532_encoding_oof_summary.csv", index=False)

    candidate_pool = pd.concat(
        [
            ablation_cv.assign(source="feature_ablation"),
            encoding_cv.assign(source="encoding"),
        ],
        ignore_index=True,
    )
    candidate_pool = candidate_pool[
        (candidate_pool["pred_std"] >= 0.18)
        & (candidate_pool["pred_endpoint_0_count"] >= 3)
        & (candidate_pool["pred_endpoint_1_count"] >= 3)
    ].sort_values("mean_mae")
    top_candidates = candidate_pool.drop_duplicates(["feature_set", "encoding"]).head(3)

    best_rows = []
    best_oofs = []
    test_diff_rows = []
    candidate_preds = {}
    for rank, row in enumerate(top_candidates.itertuples(index=False), start=1):
        fs_name = row.feature_set
        enc_name = row.encoding
        name = f"rank{rank}_{fs_name}__{enc_name}"
        summary, _, oof, _ = evaluate_cv(train_df, name, fs_name, enc_name)
        full_pred = fit_full_train_predict(train_df, test_df, fs_name, enc_name)
        file_stem = f"v532_candidate_{rank}_{fs_name.lower()}_{enc_name.lower()}"
        write_submission(sample_submission, full_pred, SUBMISSIONS_DIR / f"{file_stem}.csv")
        best_rows.append({**summary, "rank": rank, "submission_file": f"{file_stem}.csv"})
        best_oofs.append(oof.assign(rank=rank))
        test_diff_rows.append(submission_diff("baseline_sentinel99", baseline_full_pred, name, full_pred))
        candidate_preds[name] = full_pred
        print(f"best combo {rank}: {fs_name} + {enc_name} = {summary['mean_mae']:.6f}")

    best_cv = add_decision_columns(pd.DataFrame(best_rows).sort_values("mean_mae"))
    best_oof = pd.concat(best_oofs, ignore_index=True)
    test_diff = pd.DataFrame(test_diff_rows)
    best_cv.to_csv(REPORTS_DIR / "v532_best_combination_cv.csv", index=False, encoding="utf-8-sig")
    best_oof.to_csv(REPORTS_DIR / "v532_best_combination_oof.csv", index=False)
    test_diff.to_csv(REPORTS_DIR / "v532_best_combination_test_diff.csv", index=False)

    final_synthesis = build_synthesis(ablation_cv, encoding_cv, best_cv, test_diff)
    (REPORTS_DIR / "v532_final_synthesis.md").write_text(final_synthesis, encoding="utf-8-sig")

    print("\n=== v5.3.2 feature ablation ===")
    print(ablation_cv[["feature_set", "mean_mae", "improvement_vs_baseline", "pred_std", "candidate_level"]].round(6).to_string(index=False))
    print("\n=== v5.3.2 encoding ===")
    print(encoding_cv[["encoding", "feature_set", "mean_mae", "improvement_vs_baseline", "pred_std", "candidate_level"]].round(6).to_string(index=False))
    print("\n=== v5.3.2 best combinations ===")
    print(best_cv[["rank", "feature_set", "encoding", "mean_mae", "improvement_vs_baseline", "submission_file", "candidate_level"]].round(6).to_string(index=False))
    return ablation_cv, encoding_cv, best_cv


def build_synthesis(ablation_cv, encoding_cv, best_cv, test_diff):
    best_feature = ablation_cv.iloc[0]
    best_encoding = encoding_cv.iloc[0]
    best_candidate = best_cv.iloc[0]
    meaningful = best_candidate["improvement_vs_baseline"] >= 0.0005
    strong = best_candidate["improvement_vs_baseline"] >= 0.001
    if strong:
        final_decision = "strong submission candidate"
    elif meaningful:
        final_decision = "submission candidate"
    elif best_candidate["improvement_vs_baseline"] > 0:
        final_decision = "micro candidate; 제출 여부는 보수적으로 판단"
    else:
        final_decision = "제출 보류; 기존 sentinel99 유지"

    ablation_table = ablation_cv[
        ["feature_set", "mean_mae", "improvement_vs_baseline", "pred_std", "candidate_level"]
    ].to_string(index=False)
    encoding_table = encoding_cv[
        ["encoding", "feature_set", "mean_mae", "improvement_vs_baseline", "pred_std", "candidate_level"]
    ].to_string(index=False)
    best_table = best_cv[
        ["rank", "feature_set", "encoding", "mean_mae", "improvement_vs_baseline", "submission_file", "candidate_level"]
    ].to_string(index=False)
    diff_table = test_diff.to_string(index=False)

    return f"""# v5.3.2 final micro synthesis

## 1. Feature ablation 결론
기준 CV MAE는 {V53_SENTINEL99_CV:.6f}이고, 재현 baseline은 v5.3 sentinel99와 같은 설정입니다.

가장 좋은 feature set은 `{best_feature['feature_set']}`이며 CV MAE는 {best_feature['mean_mae']:.6f}, 기준 대비 개선폭은 {best_feature['improvement_vs_baseline']:.6f}입니다.

```text
{ablation_table}
```

## 2. RBF 거리 구조를 해친 파생변수 여부
파생변수 제거/정리 실험의 목적은 feature를 더 늘리는 것이 아니라 RBF 거리 구조에 불필요한 축이 있는지 보는 것입니다.
`{best_feature['feature_set']}` 결과를 기준으로 보면, CV 개선폭이 0.0003 이상이면 의미 있는 정리 후보로 볼 수 있고, 그보다 작으면 baseline S2 전체가 충분히 안정적이라고 해석합니다.

## 3. Categorical encoding 결론
가장 좋은 encoding은 `{best_encoding['encoding']}`이며 CV MAE는 {best_encoding['mean_mae']:.6f}, 기준 대비 개선폭은 {best_encoding['improvement_vs_baseline']:.6f}입니다.

```text
{encoding_table}
```

all-onehot 계열이 좋아지면 category 간 임의 순서보다 분리 표현이 RBF에 적합하다는 뜻이고, current가 유지되면 기존 단순 encoding만으로도 거리 구조가 충분하다는 해석입니다.

## 4. 최종 조합 후보
```text
{best_table}
```

baseline sentinel99 제출과의 test prediction diff는 아래와 같습니다.

```text
{diff_table}
```

## 5. 제출 판단
최종 판단: {final_decision}.

CV 개선폭이 0.0005 이상이면 제출 후보, 0.001 이상이면 strong 후보로 봅니다. 개선폭이 0.0001~0.0003이면 micro 후보이며, prediction std 축소나 endpoint 약화가 있으면 제출을 보류합니다.

0.12999 진입 가능성은 CV 개선폭과 test prediction diff가 함께 충분할 때만 주장할 수 있습니다. diff가 지나치게 작으면 LB도 거의 같을 가능성이 높고, diff가 크지만 CV 개선이 작으면 과적합 가능성을 함께 표시해야 합니다.

## 6. PPT 반영 문장
- 추가 성능 개선을 위해 feature를 무작정 늘리기보다, RBF-SVR의 거리 구조를 고려해 파생변수와 categorical encoding을 재검토했습니다.
- RBF 기반 모델에서는 단순 feature 추가보다 feature representation이 중요하며, 불필요한 파생변수는 거리 구조를 흐릴 수 있습니다.
- 최종적으로 성능, 해석 가능성, 제출 안정성을 함께 고려해 기존 sentinel99 모델 유지 또는 개선 후보 1개만 추가 제출 대상으로 선정했습니다.
"""


if __name__ == "__main__":
    run_v532()
