from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ID_COL, REPORTS_DIR, TARGET, TRAIN_PATH
from src.models_v81 import add_analysis_columns, get_rbf_oof


EXP_ID = f"v82_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def mw_group(values, scheme):
    if scheme == "current_fine":
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
    if scheme == "coarse_extreme":
        return np.select(
            [values.isna(), values <= 6, values.between(7, 10), values == 11, values >= 12],
            ["missing", "<=6", "7_10", "11", ">=12"],
            default=">=12",
        )
    if scheme == "high_11plus":
        return np.select(
            [values.isna(), values <= 6, values.between(7, 10), values >= 11],
            ["missing", "<=6", "7_10", "11plus"],
            default="11plus",
        )
    if scheme == "high_12plus":
        return np.select(
            [values.isna(), values <= 6, values.between(7, 11), values >= 12],
            ["missing", "<=6", "7_11", "12plus"],
            default="12plus",
        )
    if scheme == "low_7_high_11":
        return np.select(
            [values.isna(), values <= 7, values.between(8, 10), values >= 11],
            ["missing", "<=7", "8_10", "11plus"],
            default="11plus",
        )
    if scheme == "low_6_mid_7_8_high_9plus":
        return np.select(
            [values.isna(), values <= 6, values.between(7, 8), values.between(9, 10), values >= 11],
            ["missing", "<=6", "7_8", "9_10", "11plus"],
            default="11plus",
        )
    raise ValueError(f"Unknown scheme: {scheme}")


def eta_squared(groups, y):
    overall = np.mean(y)
    ss_between = 0.0
    ss_total = np.sum((y - overall) ** 2)
    for _, idx in groups.items():
        vals = y[idx]
        ss_between += len(vals) * (np.mean(vals) - overall) ** 2
    return float(ss_between / ss_total) if ss_total else np.nan


def cut_scheme_summary(df, oof):
    merged = df.merge(oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    merged["pred100"] = merged["oof_pred"] * 100.0
    merged["residual100"] = merged["y100"] - merged["pred100"]
    schemes = [
        "current_fine",
        "coarse_extreme",
        "high_11plus",
        "high_12plus",
        "low_7_high_11",
        "low_6_mid_7_8_high_9plus",
    ]
    rows = []
    detail_rows = []
    for scheme in schemes:
        labels = mw_group(merged["mean_working"], scheme)
        tmp = merged.assign(mw_scheme=scheme, mw_group_candidate=labels)
        group_indices = {
            key: np.flatnonzero(tmp["mw_group_candidate"].to_numpy() == key)
            for key in pd.unique(tmp["mw_group_candidate"])
        }
        y = tmp["y100"].to_numpy()
        resid = tmp["residual100"].to_numpy()
        summary = {
            "exp_id": EXP_ID,
            "scheme": scheme,
            "n_groups": int(tmp["mw_group_candidate"].nunique()),
            "min_group_count": int(tmp["mw_group_candidate"].value_counts().min()),
            "eta2_y100": eta_squared(group_indices, y),
            "eta2_residual100": eta_squared(group_indices, resid),
            "between_group_y100_range": float(tmp.groupby("mw_group_candidate")["y100"].mean().max() - tmp.groupby("mw_group_candidate")["y100"].mean().min()),
            "between_group_residual_range": float(tmp.groupby("mw_group_candidate")["residual100"].mean().max() - tmp.groupby("mw_group_candidate")["residual100"].mean().min()),
        }
        rows.append(summary)
        group_detail = tmp.groupby("mw_group_candidate").agg(
            count=("y100", "size"),
            y100_mean=("y100", "mean"),
            y100_std=("y100", "std"),
            residual100_mean=("residual100", "mean"),
            residual100_mae=("residual100", lambda s: np.mean(np.abs(s))),
            pred100_mean=("pred100", "mean"),
        ).reset_index()
        group_detail.insert(0, "scheme", scheme)
        group_detail.insert(0, "exp_id", EXP_ID)
        detail_rows.append(group_detail)
    return pd.DataFrame(rows).sort_values("eta2_y100", ascending=False), pd.concat(detail_rows, ignore_index=True)


def interaction_deep_dive(df, oof, scheme="coarse_extreme"):
    merged = df.merge(oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    merged["pred100"] = merged["oof_pred"] * 100.0
    merged["residual100"] = merged["y100"] - merged["pred100"]
    merged["mw_cut"] = mw_group(merged["mean_working"], scheme)
    factors = ["sleep_pattern", "activity", "smoke_status", "medical_history", "family_medical_history", "edu_level"]
    rows = []
    for factor in factors:
        tmp = merged.copy()
        tmp[factor] = tmp[factor].astype("object").fillna("__MISSING__")
        global_y = tmp["y100"].mean()
        global_r = tmp["residual100"].mean()
        mw_mean_y = tmp.groupby("mw_cut")["y100"].mean()
        f_mean_y = tmp.groupby(factor)["y100"].mean()
        mw_mean_r = tmp.groupby("mw_cut")["residual100"].mean()
        f_mean_r = tmp.groupby(factor)["residual100"].mean()
        for (mw, fv), group in tmp.groupby(["mw_cut", factor]):
            if len(group) < 20:
                continue
            expected_y = mw_mean_y.loc[mw] + f_mean_y.loc[fv] - global_y
            expected_r = mw_mean_r.loc[mw] + f_mean_r.loc[fv] - global_r
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "scheme": scheme,
                    "factor": factor,
                    "mw_cut": mw,
                    "factor_value": fv,
                    "count": len(group),
                    "y100_mean": float(group["y100"].mean()),
                    "expected_additive_y100": float(expected_y),
                    "interaction_excess_y100": float(group["y100"].mean() - expected_y),
                    "residual100_mean": float(group["residual100"].mean()),
                    "expected_additive_residual100": float(expected_r),
                    "interaction_excess_residual100": float(group["residual100"].mean() - expected_r),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_interaction_excess_y100"] = out["interaction_excess_y100"].abs()
    out["abs_interaction_excess_residual100"] = out["interaction_excess_residual100"].abs()
    return out.sort_values(["abs_interaction_excess_y100", "count"], ascending=[False, False])


def rule_candidate_table(interactions):
    strong = interactions[
        (interactions["count"] >= 30)
        & (interactions["abs_interaction_excess_y100"] >= 5)
    ].copy()
    strong["rule_text"] = (
        "if mean_working "
        + strong["mw_cut"].astype(str)
        + " and "
        + strong["factor"].astype(str)
        + " == "
        + strong["factor_value"].astype(str)
        + " then interaction shift ~ "
        + strong["interaction_excess_y100"].round(2).astype(str)
        + " y100"
    )
    return strong.sort_values("abs_interaction_excess_y100", ascending=False)


def build_synthesis(cut_summary, cut_detail, interactions, rules):
    best_cut = cut_summary.iloc[0]
    top_rules = rules.head(15)
    top_interactions = interactions[interactions["count"] >= 30].head(15)
    detail_best = cut_detail[cut_detail["scheme"].eq(best_cut["scheme"])]
    return f"""# v8.2 mean_working cut and interaction deep dive

## 1. mean_working cut 조정 필요성
가장 y100 분리도가 큰 cut scheme은 `{best_cut['scheme']}`입니다.

```text
{cut_summary.to_string(index=False)}
```

best scheme의 그룹별 평균:

```text
{detail_best.to_string(index=False)}
```

해석: 제출용 RBF는 raw sentinel numeric을 쓰고 있어서 cut을 직접 바꿀 필요는 크지 않습니다. 다만 해석용 score rule로는 `<=6`, `7~10/11`, `>=12`, `missing`을 분리하는 구조가 가장 자연스럽습니다.

## 2. mean_working x categorical interaction 후보
count >= 30이며 interaction excess가 큰 조합입니다.

```text
{top_interactions[['factor', 'mw_cut', 'factor_value', 'count', 'y100_mean', 'expected_additive_y100', 'interaction_excess_y100', 'residual100_mean']].to_string(index=False)}
```

## 3. 숨은 score rule 후보 문장
```text
{top_rules[['rule_text', 'count', 'y100_mean', 'residual100_mean']].to_string(index=False)}
```

## 4. 결론
- mean_working cut은 제출 모델에서 직접 조정하기보다, RBF가 raw numeric sentinel 공간에서 암묵적으로 학습하도록 두는 편이 안전합니다.
- 해석 관점에서는 `<=6`은 낮은 stress score 군, `11`과 `>=12`는 높은 stress score 군으로 뚜렷하게 분리됩니다.
- interaction 후보는 `medical_history`, `family_medical_history`, `smoke_status`, `sleep_pattern`과 결합될 때 더 강해집니다.
- 따라서 숨은 생성식은 단일 mean_working 점수표가 아니라 mean_working 구간과 건강/수면/흡연/가족력 항목의 조합 score item이 섞인 구조일 가능성이 큽니다.
"""


def run_v82():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    df = add_analysis_columns(train_df)
    oof = get_rbf_oof(train_df)

    cut_summary, cut_detail = cut_scheme_summary(df, oof)
    cut_summary.to_csv(REPORTS_DIR / "v82_mean_working_cut_scheme_summary.csv", index=False, encoding="utf-8-sig")
    cut_detail.to_csv(REPORTS_DIR / "v82_mean_working_cut_scheme_detail.csv", index=False, encoding="utf-8-sig")

    best_scheme = cut_summary.iloc[0]["scheme"]
    interactions = interaction_deep_dive(df, oof, scheme=best_scheme)
    interactions.to_csv(REPORTS_DIR / "v82_mean_working_interaction_deep_dive.csv", index=False, encoding="utf-8-sig")
    rules = rule_candidate_table(interactions)
    rules.to_csv(REPORTS_DIR / "v82_hidden_score_rule_candidates.csv", index=False, encoding="utf-8-sig")

    synthesis = build_synthesis(cut_summary, cut_detail, interactions, rules)
    (REPORTS_DIR / "v82_mean_working_rule_synthesis.md").write_text(synthesis, encoding="utf-8-sig")

    print("\n=== v8.2 cut summary ===")
    print(cut_summary.round(6).to_string(index=False))
    print("\n=== v8.2 top rules ===")
    print(rules[["rule_text", "count", "y100_mean", "residual100_mean"]].head(12).round(4).to_string(index=False))
    return cut_summary


if __name__ == "__main__":
    run_v82()
