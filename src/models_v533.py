from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVR

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
from src.models_v54 import apply_grid_postprocess


EXP_ID = f"v533_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BASE_C = 3.963530707518144
BASE_GAMMA = 1.0631617004546035
BASE_CV = 0.13417
BASELINE_SUBMISSION = SUBMISSIONS_DIR / "v532_candidate_1_fs8_no_bp_keep_bmi_metabolic_enc4_no_ordinal_except_binary.csv"
N_SPLITS = 10
RANDOM_STATE = 42


def multiply_block(X, weight=1.0):
    return X * weight


class V533FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, sentinel_value=99.0):
        self.sentinel_value = sentinel_value

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

        X_df["mean_working"] = X_df["mean_working"].fillna(float(self.sentinel_value))
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)
        ratio = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        product = X_df["cholesterol"] * X_df["glucose"]
        X_df["glucose_cholesterol_ratio"] = ratio
        X_df["cholesterol_glucose_product"] = product
        X_df["log_glucose_cholesterol_ratio"] = np.log(np.maximum(ratio, 1e-12))
        X_df["log_cholesterol_glucose_product"] = np.log1p(np.maximum(product, 0))
        X_df["gender_code"] = X_df["gender_cat"].map({"F": 0, "M": 1}).astype(float)
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])


def rbf_estimator(c_mult=1.0, gamma_mult=1.0):
    return SVR(
        kernel="rbf",
        C=BASE_C * c_mult,
        gamma=BASE_GAMMA * gamma_mult,
        epsilon=0.0,
        shrinking=True,
        cache_size=500,
    )


def numeric_columns(metabolic_mode):
    cols = [
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
        "gender_code",
    ]
    if metabolic_mode == "current_ratio_product":
        cols += ["glucose_cholesterol_ratio", "cholesterol_glucose_product"]
    elif metabolic_mode == "ratio_only":
        cols += ["glucose_cholesterol_ratio"]
    elif metabolic_mode == "product_only":
        cols += ["cholesterol_glucose_product"]
    elif metabolic_mode == "log_ratio_log_product":
        cols += ["log_glucose_cholesterol_ratio", "log_cholesterol_glucose_product"]
    elif metabolic_mode == "log_product_plus_raw_ratio":
        cols += ["glucose_cholesterol_ratio", "log_cholesterol_glucose_product"]
    elif metabolic_mode == "no_metabolic_derived":
        pass
    else:
        raise ValueError(f"Unknown metabolic_mode: {metabolic_mode}")
    return cols


def scaler_step(scaler_name):
    if scaler_name in {"current", "robust"}:
        return RobustScaler()
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "power_yeojohnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    if scaler_name == "quantile_normal_x_only":
        return QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown scaler_name: {scaler_name}")


def make_pipeline(config):
    numeric = numeric_columns(config.get("metabolic_mode", "current_ratio_product"))
    categorical = [
        "activity_cat",
        "sleep_pattern_cat",
        "edu_level_cat",
        "smoke_status_cat",
        "medical_history_cat",
        "family_medical_history_cat",
    ]
    cat_weight = float(config.get("cat_weight", 1.0))
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", scaler_step(config.get("scaler", "current"))),
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
                        (
                            "weight",
                            FunctionTransformer(
                                multiply_block,
                                kw_args={"weight": cat_weight},
                                accept_sparse=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return Pipeline(
        steps=[
            ("features", V533FeatureEngineer(config.get("sentinel_value", 99.0))),
            ("preprocess", preprocessor),
            ("dense", DenseTransformer()),
            (
                "model",
                TargetModeRegressor(
                    rbf_estimator(config.get("c_mult", 1.0), config.get("gamma_mult", 1.0)),
                    target_mode="raw",
                ),
            ),
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
    return {
        name: float(mean_absolute_error(y[mask], pred[mask])) if np.any(mask) else np.nan
        for name, mask in bins.items()
    }


def summarize_prediction(y, raw_pred, round_pred):
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


def evaluate_cv(train_df, name, config):
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    base_pipeline = make_pipeline(config)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(base_pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold

    round_oof = apply_grid_postprocess(raw_oof, "round2")
    for fold in range(1, N_SPLITS + 1):
        mask = folds == fold
        fold_rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "fold": fold,
                "mae": float(mean_absolute_error(y[mask], round_oof[mask])),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=1)),
        "base_cv": BASE_CV,
        "improvement_vs_v532_base": float(BASE_CV - fold_df["mae"].mean()),
        "C": BASE_C * config.get("c_mult", 1.0),
        "gamma": BASE_GAMMA * config.get("gamma_mult", 1.0),
        **config,
        **summarize_prediction(y, raw_oof, round_oof),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": name,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": round_oof,
        }
    )
    return summary, fold_df, oof


def fit_full_predict(train_df, test_df, config):
    model = make_pipeline(config)
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


def add_decision(df):
    out = df.copy()
    out["candidate_level"] = np.select(
        [
            out["improvement_vs_v532_base"] >= 0.001,
            out["improvement_vs_v532_base"] >= 0.0005,
            out["improvement_vs_v532_base"] > 0.0001,
            out["improvement_vs_v532_base"] > 0,
        ],
        ["strong_candidate", "submission_candidate", "micro_candidate", "tiny_micro_candidate"],
        default="hold",
    )
    out["std_warning"] = out["pred_std"] < 0.18
    out["endpoint_warning"] = (out["pred_endpoint_0_count"] < 3) | (out["pred_endpoint_1_count"] < 3)
    out["extreme_mae_warning"] = (
        (out["mae_y_le_0_1"] > 0.23) | (out["mae_y_ge_0_9"] > 0.27)
    )
    return out


def write_submission(sample_submission, pred, path):
    sub = sample_submission.copy()
    sub[TARGET] = pred
    sub.to_csv(path, index=False)


def base_config(**updates):
    config = {
        "c_mult": 1.0,
        "gamma_mult": 1.0,
        "scaler": "current",
        "metabolic_mode": "current_ratio_product",
        "cat_weight": 1.0,
        "sentinel_value": 99.0,
    }
    config.update(updates)
    return config


def run_group(train_df, group_name, candidates):
    rows = []
    folds = []
    oofs = []
    for name, config in candidates:
        summary, fold_df, oof = evaluate_cv(train_df, name, config)
        summary["group"] = group_name
        rows.append(summary)
        folds.append(fold_df.assign(group=group_name))
        oofs.append(oof.assign(group=group_name))
        print(f"{group_name} {name}: {summary['mean_mae']:.6f}")
    return add_decision(pd.DataFrame(rows).sort_values("mean_mae")), pd.concat(folds), pd.concat(oofs)


def run_v533():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    base_name = "v532_candidate1_reproduce_fs8_enc4"
    base_summary, _, base_oof = evaluate_cv(train_df, base_name, base_config())
    base_pred = fit_full_predict(train_df, test_df, base_config())
    write_submission(sample_submission, base_pred, SUBMISSIONS_DIR / "v533_base_reproduce.csv")
    existing_diff = {}
    if BASELINE_SUBMISSION.exists():
        existing_pred = pd.read_csv(BASELINE_SUBMISSION)[TARGET].to_numpy()
        existing_diff = submission_diff(BASELINE_SUBMISSION.name, existing_pred, "v533_base_reproduce", base_pred)
    else:
        existing_diff = {"left": str(BASELINE_SUBMISSION), "right": "v533_base_reproduce", "status": "missing"}
    pd.DataFrame([base_summary]).to_csv(REPORTS_DIR / "v533_base_reproduce_cv.csv", index=False, encoding="utf-8-sig")
    base_oof.to_csv(REPORTS_DIR / "v533_base_reproduce_oof.csv", index=False)

    cg_candidates = []
    for c_mult in [0.85, 1.0, 1.15]:
        for gamma_mult in [0.85, 1.0, 1.15]:
            cg_candidates.append(
                (
                    f"cg_c{c_mult:g}_g{gamma_mult:g}",
                    base_config(c_mult=c_mult, gamma_mult=gamma_mult),
                )
            )
    cg_cv, cg_folds, cg_oof = run_group(train_df, "c_gamma", cg_candidates)
    cg_cv.to_csv(REPORTS_DIR / "v533_c_gamma_tuning_cv.csv", index=False, encoding="utf-8-sig")
    cg_folds.to_csv(REPORTS_DIR / "v533_c_gamma_tuning_fold_results.csv", index=False)
    cg_cv.to_csv(REPORTS_DIR / "v533_c_gamma_tuning_oof_summary.csv", index=False, encoding="utf-8-sig")

    scaler_candidates = [
        ("SCALE0_current", base_config(scaler="current")),
        ("SCALE1_standard", base_config(scaler="standard")),
        ("SCALE2_robust", base_config(scaler="robust")),
        ("SCALE3_power_yeojohnson", base_config(scaler="power_yeojohnson")),
        ("SCALE4_quantile_normal_X_only", base_config(scaler="quantile_normal_x_only")),
    ]
    scaler_cv, _, _ = run_group(train_df, "scaler", scaler_candidates)
    scaler_cv.to_csv(REPORTS_DIR / "v533_scaler_cv.csv", index=False, encoding="utf-8-sig")
    scaler_cv.to_csv(REPORTS_DIR / "v533_scaler_oof_summary.csv", index=False, encoding="utf-8-sig")

    metabolic_candidates = [
        ("MET0_current_ratio_product", base_config(metabolic_mode="current_ratio_product")),
        ("MET1_ratio_only", base_config(metabolic_mode="ratio_only")),
        ("MET2_product_only", base_config(metabolic_mode="product_only")),
        ("MET3_log_ratio_log_product", base_config(metabolic_mode="log_ratio_log_product")),
        ("MET4_log_product_plus_raw_ratio", base_config(metabolic_mode="log_product_plus_raw_ratio")),
        ("MET5_no_metabolic_derived", base_config(metabolic_mode="no_metabolic_derived")),
    ]
    met_cv, _, _ = run_group(train_df, "metabolic", metabolic_candidates)
    met_cv.to_csv(REPORTS_DIR / "v533_metabolic_representation_cv.csv", index=False, encoding="utf-8-sig")
    met_cv.to_csv(REPORTS_DIR / "v533_metabolic_oof_summary.csv", index=False, encoding="utf-8-sig")

    catw_candidates = [
        ("CATW0_1.00", base_config(cat_weight=1.0)),
        ("CATW1_0.75", base_config(cat_weight=0.75)),
        ("CATW2_0.50", base_config(cat_weight=0.50)),
        ("CATW3_1.25", base_config(cat_weight=1.25)),
    ]
    catw_cv, _, _ = run_group(train_df, "cat_weight", catw_candidates)
    catw_cv.to_csv(REPORTS_DIR / "v533_categorical_block_weight_cv.csv", index=False, encoding="utf-8-sig")
    catw_cv.to_csv(REPORTS_DIR / "v533_categorical_block_weight_oof_summary.csv", index=False, encoding="utf-8-sig")

    sentinel_candidates = [
        ("SENT99", base_config(sentinel_value=99.0)),
        ("SENT150", base_config(sentinel_value=150.0)),
        ("SENT50", base_config(sentinel_value=50.0)),
    ]
    sent_cv, _, _ = run_group(train_df, "sentinel", sentinel_candidates)
    sent_cv.to_csv(REPORTS_DIR / "v533_sentinel_micro_cv.csv", index=False, encoding="utf-8-sig")
    sent_diffs = []
    sent_preds = {}
    for name, config in sentinel_candidates:
        pred = fit_full_predict(train_df, test_df, config)
        sent_preds[name] = pred
        sent_diffs.append(submission_diff("SENT99", sent_preds.get("SENT99", base_pred), name, pred))
    pd.DataFrame(sent_diffs).to_csv(REPORTS_DIR / "v533_sentinel_micro_test_diff.csv", index=False)

    all_cv = pd.concat(
        [
            cg_cv,
            scaler_cv,
            met_cv,
            catw_cv,
            sent_cv,
        ],
        ignore_index=True,
    )
    eligible = all_cv[
        (all_cv["improvement_vs_v532_base"] > 0)
        & (~all_cv["std_warning"])
        & (~all_cv["endpoint_warning"])
        & (~all_cv["extreme_mae_warning"])
    ].sort_values(["mean_mae", "candidate"])
    top = eligible.drop_duplicates("candidate").head(3)
    final_rows = []
    final_diffs = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        config = {
            "c_mult": row.c_mult,
            "gamma_mult": row.gamma_mult,
            "scaler": row.scaler,
            "metabolic_mode": row.metabolic_mode,
            "cat_weight": row.cat_weight,
            "sentinel_value": row.sentinel_value,
        }
        pred = fit_full_predict(train_df, test_df, config)
        slug = str(row.candidate).lower().replace(".", "p").replace("_", "_")
        descriptive = f"v533_candidate_{rank}_{slug}_fs8_enc4.csv"
        generic = f"v533_candidate_{rank}.csv"
        write_submission(sample_submission, pred, SUBMISSIONS_DIR / descriptive)
        write_submission(sample_submission, pred, SUBMISSIONS_DIR / generic)
        final_row = row._asdict()
        final_row["rank"] = rank
        final_row["submission_file"] = descriptive
        final_row["generic_submission_file"] = generic
        final_rows.append(final_row)
        final_diffs.append(submission_diff("v532_candidate1", base_pred, row.candidate, pred))

    final_summary = pd.DataFrame(final_rows)
    final_diff = pd.DataFrame(final_diffs)
    final_summary.to_csv(REPORTS_DIR / "v533_final_candidate_summary.csv", index=False, encoding="utf-8-sig")
    final_diff.to_csv(REPORTS_DIR / "v533_final_test_diff.csv", index=False)
    synthesis = build_synthesis(base_summary, existing_diff, cg_cv, scaler_cv, met_cv, catw_cv, sent_cv, final_summary, final_diff)
    (REPORTS_DIR / "v533_final_synthesis.md").write_text(synthesis, encoding="utf-8-sig")

    print("\n=== v5.3.3 base reproduce ===")
    print(pd.DataFrame([base_summary]).round(6).to_string(index=False))
    print("\n=== v5.3.3 final candidates ===")
    print(final_summary[["rank", "candidate", "group", "mean_mae", "improvement_vs_v532_base", "submission_file"]].round(6).to_string(index=False))
    return final_summary


def best_line(df):
    row = df.sort_values("mean_mae").iloc[0]
    return row["candidate"], row["mean_mae"], row["improvement_vs_v532_base"]


def table(df, cols):
    return df[cols].sort_values("mean_mae").to_string(index=False)


def build_synthesis(base_summary, existing_diff, cg_cv, scaler_cv, met_cv, catw_cv, sent_cv, final_summary, final_diff):
    cg_best = best_line(cg_cv)
    scaler_best = best_line(scaler_cv)
    met_best = best_line(met_cv)
    catw_best = best_line(catw_cv)
    sent_best = best_line(sent_cv)
    if final_summary.empty:
        final_decision = "추가 제출 후보 없음. v5.3.2 candidate1 유지가 가장 보수적입니다."
        candidate_text = "No eligible candidate."
        diff_text = "No final test diff."
    else:
        best = final_summary.sort_values("mean_mae").iloc[0]
        if best["improvement_vs_v532_base"] >= 0.001:
            level = "strong 후보"
        elif best["improvement_vs_v532_base"] >= 0.0005:
            level = "제출 후보"
        else:
            level = "micro 후보"
        final_decision = f"`{best['submission_file']}`를 {level}로 볼 수 있습니다."
        candidate_text = final_summary[
            ["rank", "candidate", "group", "mean_mae", "improvement_vs_v532_base", "submission_file"]
        ].to_string(index=False)
        diff_text = final_diff.to_string(index=False)

    return f"""# v5.3.3 RBF representation final check

## 1. v5.3.2 best candidate 재현
v5.3.2 candidate1 재현 CV MAE는 {base_summary['mean_mae']:.6f}입니다.
기준 0.134170 근처로 재현되었으며, 기존 candidate1 제출과의 diff는 `{existing_diff}`입니다.

## 2. C/gamma narrow tuning
가장 좋은 C/gamma 후보는 `{cg_best[0]}`이고 CV MAE는 {cg_best[1]:.6f}, base 대비 개선폭은 {cg_best[2]:.6f}입니다.

```text
{table(cg_cv, ['candidate', 'C', 'gamma', 'mean_mae', 'improvement_vs_v532_base', 'pred_std', 'candidate_level'])}
```

## 3. Numeric scaler / distribution representation
가장 좋은 scaler 후보는 `{scaler_best[0]}`이고 CV MAE는 {scaler_best[1]:.6f}, 개선폭은 {scaler_best[2]:.6f}입니다.
X feature의 QuantileTransformer는 target quantile이 아니라 numeric distribution normalization 실험입니다.

```text
{table(scaler_cv, ['candidate', 'scaler', 'mean_mae', 'improvement_vs_v532_base', 'pred_std', 'candidate_level'])}
```

## 4. Metabolic representation
가장 좋은 metabolic 후보는 `{met_best[0]}`이고 CV MAE는 {met_best[1]:.6f}, 개선폭은 {met_best[2]:.6f}입니다.
ratio/product가 서로 다른 정보를 주는지, log product가 거리 구조를 안정화하는지 확인했습니다.

```text
{table(met_cv, ['candidate', 'metabolic_mode', 'mean_mae', 'improvement_vs_v532_base', 'pred_std', 'candidate_level'])}
```

## 5. One-hot block weight
가장 좋은 categorical block weight 후보는 `{catw_best[0]}`이고 CV MAE는 {catw_best[1]:.6f}, 개선폭은 {catw_best[2]:.6f}입니다.
weight를 낮춘 후보가 좋아지면 one-hot block 영향이 다소 강했다는 해석이 가능하고, 1.0이 유지되면 기존 균형이 충분하다고 봅니다.

```text
{table(catw_cv, ['candidate', 'cat_weight', 'mean_mae', 'improvement_vs_v532_base', 'pred_std', 'candidate_level'])}
```

## 6. Sentinel micro
가장 좋은 sentinel 후보는 `{sent_best[0]}`이고 CV MAE는 {sent_best[1]:.6f}, 개선폭은 {sent_best[2]:.6f}입니다.
설명 가능성 기준으로는 sentinel99를 우선 유지하고, sentinel150은 micro-gamble로만 봅니다.

```text
{table(sent_cv, ['candidate', 'sentinel_value', 'mean_mae', 'improvement_vs_v532_base', 'pred_std', 'candidate_level'])}
```

## 7. 최종 제출 후보
{final_decision}

```text
{candidate_text}
```

baseline v5.3.2 candidate1 대비 test prediction diff:

```text
{diff_text}
```

## 8. 0.12999 진입 가능성
CV 개선폭이 0.0005 이상이고 prediction std와 endpoint가 유지되는 후보라면 기존 LB 0.13023에서 0.12999 이하 진입 가능성을 기대할 근거가 있습니다.
다만 이번 실험은 LB fitting이 아니라 fold-safe local CV와 설명 가능한 representation 개선에 근거한 후보 선별입니다.

## 9. PPT 보고 문장
- 최종 단계에서는 RBF-SVR의 거리 기반 특성을 고려해 feature representation을 추가 점검했습니다.
- 범주형 변수는 one-hot으로 분리하되, one-hot block의 거리 영향이 과도하지 않은지 확인했습니다.
- 대사 관련 조합 변수는 ratio와 product가 서로 다른 정보를 제공하는지 ablation으로 확인했습니다.
- 최종 후보는 단순한 변수 추가 모델이 아니라, RBF가 해석 가능한 거리 구조를 학습하도록 feature space를 정리한 모델입니다.
"""


if __name__ == "__main__":
    run_v533()
