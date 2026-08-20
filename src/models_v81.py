from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, export_text

from src.config import ID_COL, REPORTS_DIR, TARGET, TRAIN_PATH
from src.models_v54 import apply_grid_postprocess
from src.models_v8 import feature_preprocessor, rbf_pipeline


EXP_ID = f"v81_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RANDOM_STATE = 42
N_SPLITS = 10


def mean_working_group(values):
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


def add_analysis_columns(train_df):
    df = train_df.copy()
    df["y100"] = df[TARGET] * 100.0
    df["mean_working_group"] = mean_working_group(df["mean_working"])
    df["endpoint_group"] = np.select(
        [
            df[TARGET].eq(0),
            df[TARGET].between(0.01, 0.03, inclusive="both"),
            df[TARGET].between(0.97, 0.99, inclusive="both"),
            df[TARGET].eq(1),
        ],
        ["y==0", "0.01~0.03", "0.97~0.99", "y==1"],
        default="middle",
    )
    return df


def get_rbf_oof(train_df):
    existing = REPORTS_DIR / "v8_rbf_reference_oof.csv"
    if existing.exists():
        oof = pd.read_csv(existing)
        if {"ID", "oof_pred"}.issubset(oof.columns):
            return oof[[ID_COL, TARGET, "fold", "raw_oof_pred", "oof_pred"]].copy()

    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df))
    folds = np.zeros(len(train_df), dtype=int)
    base = rbf_pipeline()
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(base)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
    return pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": apply_grid_postprocess(raw_oof, "round2"),
        }
    )


def conditional_lattice(df, group_cols):
    rows = []
    for col in group_cols:
        tmp = df.copy()
        tmp[col] = tmp[col].astype("object").fillna("__MISSING__")
        for value, group in tmp.groupby(col):
            y100 = np.round(group["y100"]).astype(int)
            counts = y100.value_counts()
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "group_col": col,
                    "group_value": value,
                    "count": len(group),
                    "y100_mean": float(group["y100"].mean()),
                    "y100_std": float(group["y100"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "y100_min": int(y100.min()),
                    "y100_max": int(y100.max()),
                    "unique_y100_count": int(y100.nunique()),
                    "top_y100": int(counts.idxmax()),
                    "top_y100_count": int(counts.max()),
                    "top_y100_share": float(counts.max() / len(group)),
                }
            )
    return pd.DataFrame(rows)


def interaction_lattice(df, pairs):
    rows = []
    for left, right in pairs:
        tmp = df.copy()
        tmp[left] = tmp[left].astype("object").fillna("__MISSING__")
        tmp[right] = tmp[right].astype("object").fillna("__MISSING__")
        global_mean = tmp["y100"].mean()
        left_mean = tmp.groupby(left)["y100"].mean()
        right_mean = tmp.groupby(right)["y100"].mean()
        for (lv, rv), group in tmp.groupby([left, right]):
            expected_additive = left_mean.loc[lv] + right_mean.loc[rv] - global_mean
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "left": left,
                    "right": right,
                    "left_value": lv,
                    "right_value": rv,
                    "count": len(group),
                    "y100_mean": float(group["y100"].mean()),
                    "y100_std": float(group["y100"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "expected_additive_y100": float(expected_additive),
                    "interaction_excess_y100": float(group["y100"].mean() - expected_additive),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["left", "right", "interaction_excess_y100"], ascending=[True, True, False]
    )


def residual_rule_fingerprint(df, oof):
    merged = df.merge(oof[[ID_COL, "oof_pred", "fold"]], on=ID_COL, how="left")
    merged["pred100"] = merged["oof_pred"] * 100.0
    merged["residual100"] = merged["y100"] - merged["pred100"]
    merged["abs_residual100"] = merged["residual100"].abs()
    rule_cols = [
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
    for col in rule_cols:
        tmp = merged.copy()
        tmp[col] = tmp[col].astype("object").fillna("__MISSING__")
        global_resid = tmp["residual100"].mean()
        global_std = tmp["residual100"].std(ddof=1)
        for value, group in tmp.groupby(col):
            n = len(group)
            std = group["residual100"].std(ddof=1) if n > 1 else np.nan
            se = std / np.sqrt(n) if n > 1 and std > 0 else np.nan
            t_like = group["residual100"].mean() / se if se and not np.isnan(se) else np.nan
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "rule_col": col,
                    "rule_value": value,
                    "count": n,
                    "residual100_mean": float(group["residual100"].mean()),
                    "residual100_median": float(group["residual100"].median()),
                    "residual100_std": float(std) if not np.isnan(std) else np.nan,
                    "residual100_mae": float(group["abs_residual100"].mean()),
                    "t_like_residual_mean": float(t_like) if not np.isnan(t_like) else np.nan,
                    "global_residual_mean": float(global_resid),
                    "global_residual_std": float(global_std),
                    "score_rule_hint": abs(group["residual100"].mean()) >= 5 and n >= 50,
                }
            )
    return pd.DataFrame(rows).sort_values("residual100_mean", ascending=False)


def fold_safe_group_offset(df, oof, group_col):
    merged = df.merge(oof[[ID_COL, "oof_pred", "fold"]], on=ID_COL, how="left")
    merged["pred100"] = merged["oof_pred"] * 100.0
    merged["residual100"] = merged["y100"] - merged["pred100"]
    y = merged[TARGET].to_numpy()
    base_pred = merged["oof_pred"].to_numpy()
    adjusted_pred100 = merged["pred100"].to_numpy().copy()
    offsets = []
    for fold in sorted(merged["fold"].unique()):
        tr = merged["fold"] != fold
        va = merged["fold"] == fold
        mapping = merged.loc[tr].groupby(group_col)["residual100"].mean()
        global_offset = merged.loc[tr, "residual100"].mean()
        fold_offsets = merged.loc[va, group_col].map(mapping).fillna(global_offset).to_numpy()
        adjusted_pred100[va.to_numpy()] = adjusted_pred100[va.to_numpy()] + fold_offsets
        offsets.append(
            pd.DataFrame(
                {
                    "exp_id": EXP_ID,
                    "group_col": group_col,
                    "fold": fold,
                    "group_value": mapping.index.astype(str),
                    "fit_offset100": mapping.to_numpy(),
                }
            )
        )
    adjusted_pred = apply_grid_postprocess(adjusted_pred100 / 100.0, "round2")
    summary = {
        "exp_id": EXP_ID,
        "group_col": group_col,
        "base_mae": float(mean_absolute_error(y, base_pred)),
        "base_round2_mae": float(mean_absolute_error(y, apply_grid_postprocess(base_pred, "round2"))),
        "adjusted_round2_mae": float(mean_absolute_error(y, adjusted_pred)),
        "improvement_vs_base_round2": float(
            mean_absolute_error(y, apply_grid_postprocess(base_pred, "round2"))
            - mean_absolute_error(y, adjusted_pred)
        ),
        "adjusted_pred_mean": float(np.mean(adjusted_pred)),
        "adjusted_pred_std": float(np.std(adjusted_pred, ddof=1)),
        "adjusted_endpoint_0_count": int(np.sum(adjusted_pred == 0)),
        "adjusted_endpoint_1_count": int(np.sum(adjusted_pred == 1)),
    }
    adjusted_oof = pd.DataFrame(
        {
            ID_COL: merged[ID_COL],
            TARGET: merged[TARGET],
            "group_col": group_col,
            "fold": merged["fold"],
            "base_oof_pred": base_pred,
            "adjusted_oof_pred": adjusted_pred,
        }
    )
    return summary, pd.concat(offsets, ignore_index=True), adjusted_oof


def shallow_rule_surrogate(df, oof):
    merged = df.merge(oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    X = merged.drop(columns=[TARGET])
    y_pred = merged["oof_pred"].to_numpy()
    prep = feature_preprocessor(dense=True)
    X_features = prep.fit_transform(X)
    feature_names = prep.named_steps["preprocess"].get_feature_names_out()
    rows = []
    rules = {}
    for depth in [2, 3, 4, 5]:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=40, random_state=RANDOM_STATE)
        tree.fit(X_features, y_pred)
        pred = tree.predict(X_features)
        rows.append(
            {
                "exp_id": EXP_ID,
                "surrogate": f"tree_depth{depth}_leaf40",
                "depth": depth,
                "mae_to_rbf_oof": float(mean_absolute_error(y_pred, pred)),
                "correlation_to_rbf_oof": float(np.corrcoef(y_pred, pred)[0, 1]),
                "r2_to_rbf_oof": float(r2_score(y_pred, pred)),
            }
        )
        rules[depth] = export_text(tree, feature_names=list(feature_names), max_depth=depth)
    return pd.DataFrame(rows), rules


def build_synthesis(lattice, interactions, residuals, offsets, surrogate):
    mw_rows = lattice[lattice["group_col"].eq("mean_working_group")].sort_values("y100_mean")
    top_resid = residuals[residuals["score_rule_hint"]].head(10)
    best_offset = offsets.sort_values("improvement_vs_base_round2", ascending=False).iloc[0]
    top_interactions = interactions[interactions["count"] >= 30].copy()
    top_interactions["abs_excess"] = top_interactions["interaction_excess_y100"].abs()
    top_interactions = top_interactions.sort_values("abs_excess", ascending=False).head(12)
    best_surrogate = surrogate.sort_values("mae_to_rbf_oof").iloc[0]
    return f"""# v8.1 hidden score rule deep dive

## 1. Mean-working conditional target lattice
```text
{mw_rows[['group_value', 'count', 'y100_mean', 'y100_std', 'y100_min', 'y100_max', 'unique_y100_count', 'top_y100', 'top_y100_share']].to_string(index=False)}
```

해석: `<=6`, `11`, `>=12`가 평균 y100에서 뚜렷하게 분리되면 근무시간 구간형 score item 가능성이 살아납니다.

## 2. Residual score-rule hints
아래는 RBF OOF가 남긴 residual100 평균이 ±5점 이상이고 표본 수가 50개 이상인 그룹입니다.

```text
{top_resid[['rule_col', 'rule_value', 'count', 'residual100_mean', 'residual100_mae', 't_like_residual_mean']].to_string(index=False)}
```

해석: 같은 방향의 residual이 큰 그룹은 RBF가 smooth하게 평균화했지만 실제 생성식에는 discrete bonus/penalty가 있을 가능성이 있습니다.

## 3. Fold-safe residual offset check
OOF residual을 다른 fold에서만 학습한 그룹 offset으로 보정했을 때 가장 큰 개선은 `{best_offset['group_col']}`입니다.

```text
{offsets.sort_values('improvement_vs_base_round2', ascending=False).to_string(index=False)}
```

이 실험은 제출용 보정이 아니라 hidden score item 검증입니다. 개선이 있으면 해당 그룹 축에 아직 설명되지 않은 offset 구조가 있다는 뜻입니다.

## 4. Interaction excess
단일 marginal 평균으로 설명되지 않는 interaction excess가 큰 조합입니다.

```text
{top_interactions[['left', 'right', 'left_value', 'right_value', 'count', 'y100_mean', 'expected_additive_y100', 'interaction_excess_y100']].to_string(index=False)}
```

해석: excess가 크고 표본 수가 충분하면 구간/상호작용 score rule 후보입니다.

## 5. Shallow rule surrogate
가장 좋은 shallow tree surrogate는 `{best_surrogate['surrogate']}`이며 RBF OOF와의 MAE는 {best_surrogate['mae_to_rbf_oof']:.6f}, correlation은 {best_surrogate['correlation_to_rbf_oof']:.6f}입니다.

```text
{surrogate.to_string(index=False)}
```

## 6. 종합
- 0~100 점수체계 가설은 유지됩니다.
- 단순 선형 변환 가설보다는, mean_working 극단부와 일부 범주 조합이 discrete score item으로 들어간 구간/상호작용형 생성식이 더 그럴듯합니다.
- 다만 residual offset은 OOF 기반 해석 도구이지 그대로 제출 보정으로 쓰면 과적합 위험이 있습니다.
- 성능이 아니라 생성식 흔적을 찾는 목적이라면 `mean_working_group`, `smoke_status`, `medical_history`, `family_medical_history`의 residual offset과 interaction excess가 가장 먼저 볼 축입니다.
"""


def run_v81():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    df = add_analysis_columns(train_df)
    oof = get_rbf_oof(train_df)

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
    lattice = conditional_lattice(df, group_cols)
    lattice.to_csv(REPORTS_DIR / "v81_conditional_lattice.csv", index=False, encoding="utf-8-sig")

    pairs = [
        ("mean_working_group", "sleep_pattern"),
        ("mean_working_group", "activity"),
        ("mean_working_group", "smoke_status"),
        ("mean_working_group", "medical_history"),
        ("mean_working_group", "family_medical_history"),
        ("glucose", "cholesterol"),
    ]
    interaction_pairs = [p for p in pairs if p[0] in df.columns and p[1] in df.columns]
    interactions = interaction_lattice(df, interaction_pairs[:-1])
    interactions.to_csv(REPORTS_DIR / "v81_interaction_lattice.csv", index=False, encoding="utf-8-sig")

    residuals = residual_rule_fingerprint(df, oof)
    residuals.to_csv(REPORTS_DIR / "v81_residual_score_rule_hints.csv", index=False, encoding="utf-8-sig")

    offset_summaries = []
    offset_details = []
    offset_oofs = []
    for col in [
        "mean_working_group",
        "sleep_pattern",
        "activity",
        "smoke_status",
        "medical_history",
        "family_medical_history",
        "edu_level",
    ]:
        summary, detail, adjusted_oof = fold_safe_group_offset(df, oof, col)
        offset_summaries.append(summary)
        offset_details.append(detail)
        offset_oofs.append(adjusted_oof)
    offset_summary = pd.DataFrame(offset_summaries)
    offset_summary.to_csv(REPORTS_DIR / "v81_fold_safe_residual_offset_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(offset_details, ignore_index=True).to_csv(
        REPORTS_DIR / "v81_fold_safe_residual_offset_values.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(offset_oofs, ignore_index=True).to_csv(REPORTS_DIR / "v81_fold_safe_residual_offset_oof.csv", index=False)

    surrogate, rules = shallow_rule_surrogate(df, oof)
    surrogate.to_csv(REPORTS_DIR / "v81_shallow_rule_surrogate_fidelity.csv", index=False, encoding="utf-8-sig")
    for depth, text in rules.items():
        (REPORTS_DIR / f"v81_shallow_tree_depth{depth}_rules.txt").write_text(text, encoding="utf-8-sig")

    synthesis = build_synthesis(lattice, interactions, residuals, offset_summary, surrogate)
    (REPORTS_DIR / "v81_hidden_score_rule_synthesis.md").write_text(synthesis, encoding="utf-8-sig")

    print("\n=== v8.1 mean_working lattice ===")
    print(
        lattice[lattice["group_col"].eq("mean_working_group")][
            ["group_value", "count", "y100_mean", "y100_std", "top_y100", "top_y100_share"]
        ].round(4).to_string(index=False)
    )
    print("\n=== v8.1 offset summary ===")
    print(offset_summary.sort_values("improvement_vs_base_round2", ascending=False).round(6).to_string(index=False))
    print("\n=== v8.1 surrogate ===")
    print(surrogate.round(6).to_string(index=False))
    return offset_summary


if __name__ == "__main__":
    run_v81()
