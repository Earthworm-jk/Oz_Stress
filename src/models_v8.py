from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config import ID_COL, REPORTS_DIR, TARGET, TRAIN_PATH
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.models_v54 import apply_grid_postprocess
from src.models_v533 import BASE_C, BASE_GAMMA, V533FeatureEngineer


EXP_ID = f"v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
N_SPLITS = 10
RANDOM_STATE = 42
EPS = 1e-4

NUMERIC_COLS = [
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
CAT_COLS = [
    "activity_cat",
    "sleep_pattern_cat",
    "edu_level_cat",
    "smoke_status_cat",
    "medical_history_cat",
    "family_medical_history_cat",
]


class TargetTransformRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, mode="raw"):
        self.estimator = estimator
        self.mode = mode

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        y_model = self._fit_transform_y(y)
        self.estimator_.fit(X, y_model)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(X)
        return self._inverse_y(pred)

    def _fit_transform_y(self, y):
        y = np.asarray(y, dtype=float)
        if self.mode == "raw":
            return y
        if self.mode == "y100":
            return y * 100.0
        if self.mode == "logit":
            clipped = np.clip(y, EPS, 1 - EPS)
            return np.log(clipped / (1 - clipped))
        if self.mode == "arcsin_sqrt":
            clipped = np.clip(y, 0, 1)
            return np.arcsin(np.sqrt(clipped))
        if self.mode == "rank_normal":
            n_quantiles = min(1000, len(y))
            self.y_transformer_ = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=RANDOM_STATE,
            )
            return self.y_transformer_.fit_transform(y.reshape(-1, 1)).ravel()
        raise ValueError(f"Unknown target transform: {self.mode}")

    def _inverse_y(self, pred):
        pred = np.asarray(pred, dtype=float)
        if self.mode == "raw":
            return pred
        if self.mode == "y100":
            return pred / 100.0
        if self.mode == "logit":
            return 1.0 / (1.0 + np.exp(-pred))
        if self.mode == "arcsin_sqrt":
            return np.square(np.sin(pred))
        if self.mode == "rank_normal":
            return self.y_transformer_.inverse_transform(pred.reshape(-1, 1)).ravel()
        raise ValueError(f"Unknown target transform: {self.mode}")


def feature_preprocessor(dense=True):
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
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CAT_COLS,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    steps = [("features", V533FeatureEngineer(99.0)), ("preprocess", preprocessor)]
    if dense:
        steps.append(("dense", DenseTransformer()))
    return Pipeline(steps)


def rbf_pipeline():
    return Pipeline(
        steps=[
            ("features", V533FeatureEngineer(99.0)),
            ("preprocess", feature_preprocessor(dense=False).named_steps["preprocess"]),
            ("dense", DenseTransformer()),
            (
                "model",
                TargetModeRegressor(
                    SVR(
                        kernel="rbf",
                        C=BASE_C,
                        gamma=BASE_GAMMA,
                        epsilon=0.0,
                        shrinking=True,
                        cache_size=500,
                    ),
                    target_mode="raw",
                ),
            ),
        ]
    )


def ridge_probe_pipeline(mode):
    return Pipeline(
        steps=[
            ("features", V533FeatureEngineer(99.0)),
            ("preprocess", feature_preprocessor(dense=False).named_steps["preprocess"]),
            ("dense", DenseTransformer()),
            ("model", TargetTransformRegressor(Ridge(alpha=1.0), mode=mode)),
        ]
    )


def evaluate_pipeline(train_df, name, pipeline):
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
    pred = apply_grid_postprocess(raw_oof, "round2")
    for fold in range(1, N_SPLITS + 1):
        mask = folds == fold
        rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "fold": fold,
                "mae": float(mean_absolute_error(y[mask], pred[mask])),
            }
        )
    fold_df = pd.DataFrame(rows)
    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=1)),
        "raw_mae": float(mean_absolute_error(y, raw_oof)),
        "clip_round2_mae": float(mean_absolute_error(y, pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "endpoint_0_count": int(np.sum(pred == 0)),
        "endpoint_1_count": int(np.sum(pred == 1)),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": name,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": pred,
        }
    )
    return summary, fold_df, oof


def lattice_analysis(train_df):
    y = train_df[TARGET].to_numpy()
    y100 = y * 100.0
    y100_round = np.round(y100)
    off_grid = np.abs(y100 - y100_round)
    counts = pd.Series(y100_round.astype(int)).value_counts().sort_index()
    lattice = pd.DataFrame(
        {
            "exp_id": [EXP_ID],
            "n_rows": [len(train_df)],
            "target_min": [float(y.min())],
            "target_max": [float(y.max())],
            "unique_target_count": [int(pd.Series(y).nunique())],
            "unique_y100_count": [int(counts.size)],
            "missing_y100_values_count": [int(101 - counts.reindex(range(101), fill_value=0).astype(bool).sum())],
            "max_grid_deviation": [float(off_grid.max())],
            "rows_exact_integer_y100": [int(np.sum(off_grid < 1e-9))],
            "top_frequency_score": [int(counts.idxmax())],
            "top_frequency_count": [int(counts.max())],
        }
    )
    freq = counts.reindex(range(101), fill_value=0).rename_axis("y100").reset_index(name="count")
    freq["exp_id"] = EXP_ID
    return lattice, freq


def duplicate_analysis(train_df):
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET]]
    key_df = train_df[feature_cols].copy()
    for col in key_df.select_dtypes(include=["object"]).columns:
        key_df[col] = key_df[col].astype("object").fillna("__MISSING__")
    key_df["__target__"] = train_df[TARGET]
    grouped = key_df.groupby(feature_cols, dropna=False)["__target__"].agg(
        count="size",
        target_nunique="nunique",
        target_min="min",
        target_max="max",
        target_std="std",
    )
    dup = grouped[grouped["count"] > 1].reset_index()
    summary = pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "duplicate_groups": int(len(dup)),
                "duplicate_rows": int(dup["count"].sum()) if len(dup) else 0,
                "conflicting_duplicate_groups": int((dup["target_nunique"] > 1).sum()) if len(dup) else 0,
                "max_duplicate_target_range": float((dup["target_max"] - dup["target_min"]).max()) if len(dup) else 0.0,
            }
        ]
    )
    return summary, dup


def residual_fingerprint(train_df, rbf_oof):
    df = train_df.copy()
    df = df.merge(rbf_oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    df["y100"] = df[TARGET] * 100.0
    df["pred100"] = df["oof_pred"] * 100.0
    df["residual100"] = df["y100"] - df["pred100"]
    df["abs_residual100"] = df["residual100"].abs()
    df["residual100_rounded"] = np.round(df["residual100"]).astype(int)
    df["mean_working_group"] = np.select(
        [
            df["mean_working"].isna(),
            df["mean_working"] <= 6,
            df["mean_working"] == 7,
            df["mean_working"] == 8,
            df["mean_working"] == 9,
            df["mean_working"] == 10,
            df["mean_working"] == 11,
            df["mean_working"] >= 12,
        ],
        ["missing", "<=6", "7", "8", "9", "10", "11", ">=12"],
        default=">=12",
    )
    group_cols = [
        "sleep_pattern",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
        "edu_level",
        "mean_working_group",
    ]
    rows = []
    for col in group_cols:
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").fillna("__MISSING__")
        stats = tmp.groupby(col).agg(
            count=("residual100", "size"),
            residual_mean=("residual100", "mean"),
            residual_median=("residual100", "median"),
            residual_mae=("abs_residual100", "mean"),
            y100_mean=("y100", "mean"),
            pred100_mean=("pred100", "mean"),
        )
        stats = stats.reset_index().rename(columns={col: "group_value"})
        stats.insert(0, "group_col", col)
        rows.append(stats)
    group_df = pd.concat(rows, ignore_index=True)
    residual_counts = (
        df["residual100_rounded"].value_counts().sort_index().rename_axis("residual100_rounded").reset_index(name="count")
    )
    residual_counts["exp_id"] = EXP_ID
    group_df["exp_id"] = EXP_ID
    return group_df, residual_counts


def knn_diagnostic(train_df):
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    ks = [5, 10, 20]
    oof = {k: np.zeros(len(train_df), dtype=float) for k in ks}
    neighbor_std = {k: np.zeros(len(train_df), dtype=float) for k in ks}
    folds = np.zeros(len(train_df), dtype=int)
    base_features = feature_preprocessor(dense=True)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        prep = clone(base_features)
        X_tr = prep.fit_transform(X.iloc[tr_idx])
        X_va = prep.transform(X.iloc[va_idx])
        max_k = max(ks)
        nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
        nn.fit(X_tr)
        _, indices = nn.kneighbors(X_va)
        for k in ks:
            targets = y[tr_idx][indices[:, :k]]
            oof[k][va_idx] = targets.mean(axis=1)
            neighbor_std[k][va_idx] = targets.std(axis=1, ddof=1)
        folds[va_idx] = fold
    rows = []
    oof_rows = []
    for k in ks:
        pred = apply_grid_postprocess(oof[k], "round2")
        rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": f"knn_k{k}",
                "k": k,
                "mean_mae": float(mean_absolute_error(y, pred)),
                "raw_mae": float(mean_absolute_error(y, oof[k])),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "neighbor_target_std_mean": float(np.mean(neighbor_std[k])),
                "neighbor_target_std_median": float(np.median(neighbor_std[k])),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    ID_COL: train_df[ID_COL],
                    TARGET: train_df[TARGET],
                    "candidate": f"knn_k{k}",
                    "fold": folds,
                    "raw_oof_pred": oof[k],
                    "oof_pred": pred,
                    "neighbor_target_std": neighbor_std[k],
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(oof_rows, ignore_index=True)


def surrogate_rules(train_df, rbf_oof):
    X = train_df.drop(columns=[TARGET])
    y_rbf = rbf_oof["oof_pred"].to_numpy()
    prep = feature_preprocessor(dense=True)
    X_features = prep.fit_transform(X)
    feature_names = prep.named_steps["preprocess"].get_feature_names_out()
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=40, random_state=RANDOM_STATE)
    tree.fit(X_features, y_rbf)
    pred_tree = tree.predict(X_features)
    extra = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    extra.fit(X_features, y_rbf)
    pred_extra = extra.predict(X_features)
    fidelity = pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "surrogate": "decision_tree_depth4",
                "mae_to_rbf_oof": float(mean_absolute_error(y_rbf, pred_tree)),
                "correlation_to_rbf_oof": float(np.corrcoef(y_rbf, pred_tree)[0, 1]),
                "r2_to_rbf_oof": float(r2_score(y_rbf, pred_tree)),
            },
            {
                "exp_id": EXP_ID,
                "surrogate": "extra_trees_leaf5",
                "mae_to_rbf_oof": float(mean_absolute_error(y_rbf, pred_extra)),
                "correlation_to_rbf_oof": float(np.corrcoef(y_rbf, pred_extra)[0, 1]),
                "r2_to_rbf_oof": float(r2_score(y_rbf, pred_extra)),
            },
        ]
    )
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": extra.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rules = export_text(tree, feature_names=list(feature_names), max_depth=4)
    return fidelity, importances, rules


def build_synthesis(lattice, dup_summary, linear_probe, knn_summary, residual_counts, surrogate_fidelity):
    best_linear = linear_probe.sort_values("mean_mae").iloc[0]
    best_knn = knn_summary.sort_values("mean_mae").iloc[0]
    lattice_row = lattice.iloc[0]
    duplicate_row = dup_summary.iloc[0]
    residual_mode = residual_counts.sort_values("count", ascending=False).head(5)
    surrogate_best = surrogate_fidelity.sort_values("mae_to_rbf_oof").iloc[0]
    if best_knn["mean_mae"] + 0.01 < best_linear["mean_mae"]:
        knn_interpretation = (
            "kNN이 Ridge보다 뚜렷하게 좋으므로 feature space에 local smooth structure가 있다는 힌트입니다."
        )
    elif abs(best_knn["mean_mae"] - best_linear["mean_mae"]) <= 0.01:
        knn_interpretation = (
            "kNN이 Ridge와 비슷하게 약하므로 단순한 근접 평균만으로는 hidden score surface를 복원하기 어렵습니다. "
            "RBF의 이득은 nearest-neighbor averaging보다 kernel interpolation과 representation 정리에서 나온 것으로 보입니다."
        )
    else:
        knn_interpretation = (
            "kNN이 Ridge보다 나쁘므로 현재 고차원 one-hot feature space의 단순 유클리드 근접성은 target 생성식을 잘 설명하지 못합니다."
        )
    return f"""# v8 hidden score generation diagnostic

## 1. Target lattice
- y100 unique count: {int(lattice_row['unique_y100_count'])}
- max grid deviation: {lattice_row['max_grid_deviation']:.10f}
- exact integer y100 rows: {int(lattice_row['rows_exact_integer_y100'])} / {int(lattice_row['n_rows'])}
- most frequent y100 score: {int(lattice_row['top_frequency_score'])} with count {int(lattice_row['top_frequency_count'])}

해석: target이 0~100 정수 점수에서 0~1로 변환되었다는 가설을 직접 지지합니다.

## 2. Duplicate determinism
- duplicate groups: {int(duplicate_row['duplicate_groups'])}
- conflicting duplicate groups: {int(duplicate_row['conflicting_duplicate_groups'])}
- max duplicate target range: {duplicate_row['max_duplicate_target_range']:.6f}

동일 feature row에서 target이 흔들리면 hidden noise나 누락 변수가 있다는 뜻이고, 거의 흔들리지 않으면 deterministic formula 복원 가능성이 커집니다.

## 3. Linear transform probe
가장 좋은 선형 target transform은 `{best_linear['candidate']}`이며 CV MAE는 {best_linear['mean_mae']:.6f}입니다.

```text
{linear_probe[['candidate', 'mean_mae', 'pred_std', 'endpoint_0_count', 'endpoint_1_count']].sort_values('mean_mae').to_string(index=False)}
```

선형 변환 후에도 RBF 수준까지 오지 못하면, 단순 선형식을 monotonic nonlinear transform한 구조만으로는 부족하다고 봅니다.

## 4. Local neighbor consistency
가장 좋은 kNN 후보는 `{best_knn['candidate']}`이며 MAE는 {best_knn['mean_mae']:.6f}입니다.

```text
{knn_summary.to_string(index=False)}
```

{knn_interpretation}

## 5. Integer residual fingerprint
OOF RBF residual100 rounded 상위 빈도:

```text
{residual_mode.to_string(index=False)}
```

잔차가 정수 근처에 몰리면 0~100 점수와 rounding의 흔적으로 볼 수 있습니다. 특정 그룹에서 residual mean이 일정하면 숨은 offset item 후보입니다.

## 6. Surrogate formula hint
가장 fidelity가 좋은 surrogate는 `{surrogate_best['surrogate']}`이며 RBF OOF 예측과의 MAE는 {surrogate_best['mae_to_rbf_oof']:.6f}, correlation은 {surrogate_best['correlation_to_rbf_oof']:.6f}입니다.

```text
{surrogate_fidelity.to_string(index=False)}
```

## 7. 종합 해석
- target grid는 0~100 점수체계 가설을 강하게 지지합니다.
- 다만 선형 target transform probe가 RBF에 근접하지 못하면, 단순 선형 점수를 비선형 변환한 문제라기보다는 구간/상호작용/부드러운 비선형 latent score일 가능성이 큽니다.
- kNN과 surrogate 결과는 hidden score surface가 feature space의 국소 구조를 가진다는 점을 확인하는 용도입니다.
- 다음으로 수식 복원을 더 밀고 싶다면 residual group offset과 surrogate feature importance 상위 항목을 중심으로 작은 rule/threshold 후보를 만들 수 있습니다.
"""


def run_v8():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)

    lattice, freq = lattice_analysis(train_df)
    dup_summary, dup_detail = duplicate_analysis(train_df)
    lattice.to_csv(REPORTS_DIR / "v8_target_lattice_summary.csv", index=False, encoding="utf-8-sig")
    freq.to_csv(REPORTS_DIR / "v8_target_y100_frequency.csv", index=False)
    dup_summary.to_csv(REPORTS_DIR / "v8_duplicate_determinism_summary.csv", index=False, encoding="utf-8-sig")
    dup_detail.to_csv(REPORTS_DIR / "v8_duplicate_determinism_detail.csv", index=False)

    rbf_summary, rbf_folds, rbf_oof = evaluate_pipeline(train_df, "rbf_v532_candidate1_reference", rbf_pipeline())
    rbf_oof.to_csv(REPORTS_DIR / "v8_rbf_reference_oof.csv", index=False)

    linear_rows = []
    linear_folds = []
    linear_oofs = []
    for mode in ["raw", "y100", "logit", "arcsin_sqrt", "rank_normal"]:
        summary, folds, oof = evaluate_pipeline(train_df, f"ridge_{mode}", ridge_probe_pipeline(mode))
        linear_rows.append(summary)
        linear_folds.append(folds)
        linear_oofs.append(oof)
        print(f"linear probe {mode}: {summary['mean_mae']:.6f}")
    linear_probe = pd.DataFrame(linear_rows).sort_values("mean_mae")
    pd.concat(linear_folds, ignore_index=True).to_csv(REPORTS_DIR / "v8_target_transform_linear_probe_folds.csv", index=False)
    pd.concat(linear_oofs, ignore_index=True).to_csv(REPORTS_DIR / "v8_target_transform_linear_probe_oof.csv", index=False)
    linear_probe.to_csv(REPORTS_DIR / "v8_target_transform_linear_probe.csv", index=False, encoding="utf-8-sig")

    residual_groups, residual_counts = residual_fingerprint(train_df, rbf_oof)
    residual_groups.to_csv(REPORTS_DIR / "v8_rbf_residual_group_fingerprint.csv", index=False, encoding="utf-8-sig")
    residual_counts.to_csv(REPORTS_DIR / "v8_rbf_residual_integer_frequency.csv", index=False)

    knn_summary, knn_oof = knn_diagnostic(train_df)
    knn_summary.to_csv(REPORTS_DIR / "v8_knn_local_consistency.csv", index=False, encoding="utf-8-sig")
    knn_oof.to_csv(REPORTS_DIR / "v8_knn_local_consistency_oof.csv", index=False)

    surrogate_fidelity, surrogate_importance, rules = surrogate_rules(train_df, rbf_oof)
    surrogate_fidelity.to_csv(REPORTS_DIR / "v8_surrogate_fidelity.csv", index=False, encoding="utf-8-sig")
    surrogate_importance.to_csv(REPORTS_DIR / "v8_surrogate_feature_importance.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "v8_surrogate_tree_rules.txt").write_text(rules, encoding="utf-8-sig")

    overview = pd.concat(
        [
            pd.DataFrame([rbf_summary]).assign(section="rbf_reference"),
            linear_probe.assign(section="linear_probe"),
            knn_summary.rename(columns={"candidate": "candidate"}).assign(section="knn"),
        ],
        ignore_index=True,
        sort=False,
    )
    overview.to_csv(REPORTS_DIR / "v8_hidden_score_diagnostic_overview.csv", index=False, encoding="utf-8-sig")

    synthesis = build_synthesis(lattice, dup_summary, linear_probe, knn_summary, residual_counts, surrogate_fidelity)
    (REPORTS_DIR / "v8_hidden_score_synthesis.md").write_text(synthesis, encoding="utf-8-sig")

    print("\n=== v8 lattice ===")
    print(lattice.to_string(index=False))
    print("\n=== v8 linear probe ===")
    print(linear_probe[["candidate", "mean_mae", "pred_std"]].round(6).to_string(index=False))
    print("\n=== v8 knn ===")
    print(knn_summary.round(6).to_string(index=False))
    print("\n=== v8 surrogate fidelity ===")
    print(surrogate_fidelity.round(6).to_string(index=False))
    return overview


if __name__ == "__main__":
    run_v8()
