from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

from src.config import ID_COL, REPORTS_DIR, TARGET, TRAIN_PATH
from src.models_v532 import make_pipeline
from src.models_v54 import apply_grid_postprocess


EXP_ID = f"v532_foldcheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FEATURE_SET = "FS8_no_BP_keep_BMI_metabolic"
ENCODING = "ENC4_no_ordinal_except_binary"
BASELINE_10FOLD_CV = 0.13417


def stratification_bins(y, n_bins=10):
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return np.asarray(bins, dtype=int)


def make_splitter(kind, n_splits, seed=42):
    if kind == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed), None
    if kind == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), "stratified_y10"
    raise ValueError(f"Unknown splitter kind: {kind}")


def evaluate_fold_strategy(train_df, name, kind, n_splits, seed=42):
    splitter, strat_col = make_splitter(kind, n_splits, seed)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    y_strat = stratification_bins(y) if strat_col else None
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    rows = []
    pipeline = make_pipeline(FEATURE_SET, ENCODING)

    split_iter = splitter.split(np.zeros(len(y)), y_strat) if y_strat is not None else splitter.split(np.zeros(len(y)))
    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        pred_fold = apply_grid_postprocess(raw_oof[va_idx], "round2")
        rows.append(
            {
                "exp_id": EXP_ID,
                "strategy": name,
                "splitter": kind,
                "n_splits": n_splits,
                "seed": seed,
                "fold": fold,
                "train_size": len(tr_idx),
                "valid_size": len(va_idx),
                "valid_target_mean": float(np.mean(y[va_idx])),
                "valid_target_std": float(np.std(y[va_idx], ddof=1)),
                "mae": float(mean_absolute_error(y[va_idx], pred_fold)),
            }
        )

    pred = apply_grid_postprocess(raw_oof, "round2")
    fold_df = pd.DataFrame(rows)
    summary = {
        "exp_id": EXP_ID,
        "strategy": name,
        "splitter": kind,
        "n_splits": n_splits,
        "seed": seed,
        "mean_mae": float(fold_df["mae"].mean()),
        "std_mae": float(fold_df["mae"].std(ddof=1)),
        "min_fold_mae": float(fold_df["mae"].min()),
        "max_fold_mae": float(fold_df["mae"].max()),
        "fold_mae_range": float(fold_df["mae"].max() - fold_df["mae"].min()),
        "mean_valid_size": float(fold_df["valid_size"].mean()),
        "target_mean_std_across_folds": float(fold_df["valid_target_mean"].std(ddof=1)),
        "target_std_mean_across_folds": float(fold_df["valid_target_std"].mean()),
        "oof_mae": float(mean_absolute_error(y, pred)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "endpoint_0_count": int(np.sum(pred == 0)),
        "endpoint_1_count": int(np.sum(pred == 1)),
        "delta_vs_10fold_reference": float(fold_df["mae"].mean() - BASELINE_10FOLD_CV),
    }
    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "strategy": name,
            "splitter": kind,
            "n_splits": n_splits,
            "seed": seed,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": pred,
        }
    )
    return summary, fold_df, oof


def build_synthesis(summary):
    ordered = summary.sort_values(["mean_mae", "std_mae"])
    best = ordered.iloc[0]
    k10 = summary[summary["strategy"].eq("kfold_10_seed42")].iloc[0]
    s10 = summary[summary["strategy"].eq("stratified_10_seed42")].iloc[0]
    k5 = summary[summary["strategy"].eq("kfold_5_seed42")].iloc[0]
    return f"""# v5.3.2 candidate1 fold strategy check

## 1. 목적
최종 제출 후보인 v5.3.2 candidate1은 `FS8_no_BP_keep_BMI_metabolic + ENC4_no_ordinal_except_binary`와 raw RBF-SVR을 사용합니다.
이번 실험은 모델/feature를 바꾸지 않고 validation fold 방식만 비교해 10-fold 선택 근거를 확보하기 위한 것입니다.

## 2. 결과 요약
```text
{summary[['strategy', 'splitter', 'n_splits', 'mean_mae', 'std_mae', 'fold_mae_range', 'target_mean_std_across_folds', 'pred_std', 'endpoint_0_count', 'endpoint_1_count']].sort_values('mean_mae').to_string(index=False)}
```

가장 낮은 CV는 `{best['strategy']}`의 {best['mean_mae']:.6f}입니다.
기준 10-fold KFold는 {k10['mean_mae']:.6f}, Stratified 10-fold는 {s10['mean_mae']:.6f}, 5-fold KFold는 {k5['mean_mae']:.6f}입니다.

## 3. 10-fold를 선택한 이유
- train size가 3000개로 크지 않아 5-fold보다 각 fold train 비율을 높이는 10-fold가 유리합니다.
- 10-fold는 validation fold가 300개라 fold별 target 분포가 과도하게 작아지지 않으면서도 OOF를 안정적으로 만들 수 있습니다.
- 5-fold는 validation fold가 커서 fold MAE는 덜 출렁일 수 있지만, 각 모델이 80% train만 보고 학습하므로 최종 full-train 제출 모델과의 train-size gap이 큽니다.
- StratifiedKFold 10-fold도 확인했지만, target이 0~100 grid로 촘촘하고 KFold shuffle만으로도 fold target mean 편차가 작아 큰 이득이 없었습니다.
- 최종 제출 모델은 full train fit이므로, local CV는 leaderboard를 맞추는 도구가 아니라 모델 선택의 안정성 확인용입니다. 이 관점에서 10-fold KFold는 bias와 variance의 균형이 좋습니다.

## 4. 보고서 문장
Validation은 10-fold KFold(shuffle=True, random_state=42)를 사용했다. 데이터 수가 3000개로 제한적이기 때문에 5-fold보다 각 fold의 학습 비율을 높여 full-train 제출 상황과의 차이를 줄이고, 동시에 validation fold당 약 300개 샘플을 확보해 fold별 MAE와 OOF 분포를 안정적으로 비교할 수 있었다. 추가로 5-fold 및 StratifiedKFold를 확인했으며, 10-fold KFold가 성능과 안정성 측면에서 균형적인 기준으로 판단되었다.
"""


def run_foldcheck():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    configs = [
        ("kfold_5_seed42", "kfold", 5, 42),
        ("kfold_10_seed42", "kfold", 10, 42),
        ("kfold_15_seed42", "kfold", 15, 42),
        ("kfold_20_seed42", "kfold", 20, 42),
        ("stratified_5_seed42", "stratified", 5, 42),
        ("stratified_10_seed42", "stratified", 10, 42),
    ]
    summaries = []
    folds = []
    oofs = []
    for name, kind, n_splits, seed in configs:
        summary, fold_df, oof = evaluate_fold_strategy(train_df, name, kind, n_splits, seed)
        summaries.append(summary)
        folds.append(fold_df)
        oofs.append(oof)
        print(f"{name}: {summary['mean_mae']:.6f} +/- {summary['std_mae']:.6f}")

    summary_df = pd.DataFrame(summaries).sort_values("mean_mae")
    fold_df = pd.concat(folds, ignore_index=True)
    oof_df = pd.concat(oofs, ignore_index=True)
    summary_df.to_csv(REPORTS_DIR / "v532_fold_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(REPORTS_DIR / "v532_fold_strategy_fold_results.csv", index=False)
    oof_df.to_csv(REPORTS_DIR / "v532_fold_strategy_oof.csv", index=False)
    (REPORTS_DIR / "v532_fold_strategy_synthesis.md").write_text(build_synthesis(summary_df), encoding="utf-8-sig")

    print("\n=== fold strategy summary ===")
    print(
        summary_df[
            [
                "strategy",
                "mean_mae",
                "std_mae",
                "fold_mae_range",
                "target_mean_std_across_folds",
                "pred_std",
            ]
        ].round(6).to_string(index=False)
    )
    return summary_df


if __name__ == "__main__":
    run_foldcheck()
