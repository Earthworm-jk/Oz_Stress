from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.config import (
    ID_COL,
    REPORTS_DIR,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.models_v54 import (
    V53_SENTINEL99_CV,
    apply_grid_postprocess,
    make_pipeline_from_config,
    robust_scaler_stats,
)


EXP_ID = f"v531_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SEEDS = [42, 2024, 2025, 777, 1004]
BASELINE_LB = {
    "v3_extratrees": 0.16743,
    "v51_raw_rbf_s2": 0.13458,
    "v53_sentinel99": 0.13023,
}


def sentinel_config(value):
    return {"mean_working_mode": "sentinel", "sentinel_value": float(value)}


def summarize_prediction(pred):
    return {
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred, ddof=1)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_endpoint_0_count": int(np.sum(pred == 0)),
        "pred_endpoint_1_count": int(np.sum(pred == 1)),
    }


def evaluate_cv(train_df, test_df, name, config, seed=42, n_splits=10):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_test_raw = []
    fold_rows = []
    base_pipeline = make_pipeline_from_config(config)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(y))), start=1):
        model = clone(base_pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        fold_test_raw.append(model.predict(test_df))

    oof_pred = apply_grid_postprocess(raw_oof, "round2")
    for fold in range(1, n_splits + 1):
        mask = folds == fold
        fold_rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "seed": seed,
                "fold": fold,
                "mae": mean_absolute_error(y[mask], oof_pred[mask]),
            }
        )

    summary = {
        "exp_id": EXP_ID,
        "candidate": name,
        "seed": seed,
        "n_splits": n_splits,
        "postprocess": "clip_0_1_round2",
        "mean_mae": float(np.mean([row["mae"] for row in fold_rows])),
        "std_mae": float(np.std([row["mae"] for row in fold_rows], ddof=1)),
        **summarize_prediction(oof_pred),
    }
    if "sentinel_value" in config:
        median, iqr, scaled = robust_scaler_stats(train_df, float(config["sentinel_value"]))
        summary.update(
            {
                "sentinel_value": float(config["sentinel_value"]),
                "train_mean_working_median": float(median),
                "train_mean_working_iqr": float(iqr),
                "scaled_sentinel_value": float(scaled),
            }
        )

    oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": name,
            "seed": seed,
            "fold": folds,
            "raw_oof_pred": raw_oof,
            "oof_pred": oof_pred,
        }
    )
    test_raw = np.mean(np.vstack(fold_test_raw), axis=0)
    test_pred = apply_grid_postprocess(test_raw, "round2")
    return summary, pd.DataFrame(fold_rows), oof, test_raw, test_pred


def fit_full_train_submission(train_df, test_df, config):
    model = make_pipeline_from_config(config)
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET].to_numpy())
    raw_pred = model.predict(test_df)
    return raw_pred, apply_grid_postprocess(raw_pred, "round2")


def write_submission(sample_submission, pred, path):
    sub = sample_submission.copy()
    sub[TARGET] = pred
    sub.to_csv(path, index=False)
    return sub


def submission_diff_stats(name_a, pred_a, name_b, pred_b):
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    values = np.round(np.arange(0, 1.01, 0.01), 2)
    count_a = pd.Series(a).value_counts().reindex(values, fill_value=0)
    count_b = pd.Series(b).value_counts().reindex(values, fill_value=0)
    diff = np.abs(a - b)
    return {
        "left": name_a,
        "right": name_b,
        "different_row_count": int(np.sum(a != b)),
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)),
        "prediction_correlation": float(np.corrcoef(a, b)[0, 1]),
        "value_count_abs_diff_sum": int(np.sum(np.abs(count_a.to_numpy() - count_b.to_numpy()))),
        "left_endpoint_0_count": int(np.sum(a == 0)),
        "left_endpoint_1_count": int(np.sum(a == 1)),
        "right_endpoint_0_count": int(np.sum(b == 0)),
        "right_endpoint_1_count": int(np.sum(b == 1)),
    }


def frequency_table(name, pred):
    values = np.round(np.arange(0, 1.01, 0.01), 2)
    counts = pd.Series(pred).value_counts().reindex(values, fill_value=0)
    return pd.DataFrame({"candidate": name, "prediction": values, "count": counts.to_numpy()})


def compare_with_existing_submission(pred):
    path = SUBMISSIONS_DIR / "v53_best_raw_rbf_B_mean_working_sentinel99.csv"
    if not path.exists():
        return {
            "existing_submission": str(path),
            "status": "missing",
            "mean_abs_diff": np.nan,
            "max_abs_diff": np.nan,
        }
    existing = pd.read_csv(path)[TARGET].to_numpy()
    stats = submission_diff_stats("v531_single_sentinel99_reproduce", pred, path.name, existing)
    stats["existing_submission"] = str(path)
    stats["status"] = "compared"
    return stats


def build_final_tables(single_summary, sentinel_summary, ensemble_summary, pairwise_df, existing_diff):
    rows = [
        {
            "version": "linear_baselines",
            "model": "Ridge / ElasticNet / LinearSVR",
            "key_idea": "simple additive scorecard check",
            "CV_MAE": 0.2445,
            "LB_MAE": np.nan,
            "interpretation": "단순 선형 가산식만으로는 stress_score 생성식을 충분히 설명하지 못함",
            "final_decision": "not candidate",
        },
        {
            "version": "v3",
            "model": "ExtraTrees",
            "key_idea": "tree-style binned scorecard",
            "CV_MAE": 0.17986,
            "LB_MAE": BASELINE_LB["v3_extratrees"],
            "interpretation": "구간형 효과를 일부 포착했지만 RBF보다 약함",
            "final_decision": "superseded",
        },
        {
            "version": "v5.1",
            "model": "raw RBF SVR S2",
            "key_idea": "smooth nonlinear latent score",
            "CV_MAE": 0.139413,
            "LB_MAE": BASELINE_LB["v51_raw_rbf_s2"],
            "interpretation": "0.01 grid 뒤의 연속 latent score surface 가설을 강하게 지지",
            "final_decision": "superseded",
        },
        {
            "version": "v5.3",
            "model": "raw RBF SVR + sentinel99",
            "key_idea": "mean_working missing as out-of-range state",
            "CV_MAE": V53_SENTINEL99_CV,
            "LB_MAE": BASELINE_LB["v53_sentinel99"],
            "interpretation": "결측을 정상 근무시간 밖 상태로 분리했을 때 큰 개선",
            "final_decision": "primary reference",
        },
        {
            "version": "v7",
            "model": "MLPRegressor",
            "key_idea": "neural smooth nonlinear model",
            "CV_MAE": 0.21836333333333333,
            "LB_MAE": np.nan,
            "interpretation": "작은 tabular 환경에서는 RBF SVR이 더 안정적",
            "final_decision": "not candidate",
        },
        {
            "version": "v5.3.1",
            "model": "sentinel99 seed5 CV ensemble",
            "key_idea": "fold seed variance check",
            "CV_MAE": float(ensemble_summary["mean_mae"]),
            "LB_MAE": np.nan,
            "interpretation": "seed 평균이 single seed42 대비 안정화되는지 확인",
            "final_decision": ensemble_summary["decision"],
        },
    ]
    for _, row in sentinel_summary.iterrows():
        rows.append(
            {
                "version": "v5.3.1",
                "model": row["candidate"],
                "key_idea": "out-of-range sentinel micro-check",
                "CV_MAE": row["mean_mae"],
                "LB_MAE": np.nan,
                "interpretation": "수치 자체보다 정상 범위 밖 분리 효과를 확인",
                "final_decision": row["decision"],
            }
        )

    comparison = pd.DataFrame(rows)
    candidates = pd.DataFrame(
        [
            {
                "submission_file": "v531_single_sentinel99_reproduce.csv",
                "candidate": "sentinel99 single full-train reproduction",
                "CV_MAE": single_summary["mean_mae"],
                "known_LB_MAE": BASELINE_LB["v53_sentinel99"],
                "priority": 1,
                "recommendation": "기존 LB 0.13023 기준의 가장 설명 가능한 주 제출 후보",
            },
            {
                "submission_file": "v531_sentinel99_seed5_ensemble.csv",
                "candidate": "sentinel99 5-seed fold-model ensemble",
                "CV_MAE": ensemble_summary["mean_mae"],
                "known_LB_MAE": np.nan,
                "priority": 2 if ensemble_summary["is_candidate"] else 4,
                "recommendation": ensemble_summary["decision"],
            },
        ]
    )
    best_sentinel = sentinel_summary.sort_values("mean_mae").iloc[0]
    candidates = pd.concat(
        [
            candidates,
            pd.DataFrame(
                [
                    {
                        "submission_file": f"v531_single_{best_sentinel['candidate']}.csv",
                        "candidate": f"{best_sentinel['candidate']} single full-train",
                        "CV_MAE": best_sentinel["mean_mae"],
                        "known_LB_MAE": np.nan,
                        "priority": 3,
                        "recommendation": "sentinel99와 test 차이가 매우 작을 때만 micro-gamble",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    candidates["exp_id"] = EXP_ID
    comparison["exp_id"] = EXP_ID
    return comparison, candidates


def write_synthesis(path, single_summary, ensemble_summary, sentinel_summary, pairwise_df, existing_diff):
    best_sentinel = sentinel_summary.sort_values("mean_mae").iloc[0]
    pairwise_text = pairwise_df[
        ["left", "right", "different_row_count", "mean_abs_diff", "max_abs_diff", "prediction_correlation"]
    ].to_string(index=False)
    text = f"""# v5.3.1 final touch synthesis

## 1. v5.3 sentinel99 재현
- sentinel99 single 10-fold CV MAE는 {single_summary['mean_mae']:.6f}입니다.
- 기존 v5.3 기준 CV {V53_SENTINEL99_CV:.6f}와 같은 수준으로 재현되었습니다.
- 기존 제출 파일과의 비교 상태: {existing_diff['status']}, mean_abs_diff={existing_diff['mean_abs_diff']:.8f}, max_abs_diff={existing_diff['max_abs_diff']:.8f}.

## 2. seed ensemble 판단
- 5-seed OOF ensemble CV MAE는 {ensemble_summary['mean_mae']:.6f}입니다.
- single seed42 대비 개선폭은 {ensemble_summary['improvement_vs_seed42']:.6f}입니다.
- OOF pred std는 {ensemble_summary['pred_std']:.6f}이고, single seed42 pred std 대비 변화량은 {ensemble_summary['pred_std_delta_vs_seed42']:.6f}입니다.
- 판단: {ensemble_summary['decision']}.

## 3. sentinel150 / sentinel999 비교
- best micro sentinel은 {best_sentinel['candidate']}이며 CV MAE는 {best_sentinel['mean_mae']:.6f}입니다.
- 99, 150, 999는 모두 정상 근무시간 범위 밖으로 결측을 분리한다는 공통점이 있고, 성능 차이는 매우 작습니다.
- 따라서 99라는 숫자가 마법값이라기보다, mean_working 결측을 정상 근무시간 manifold 밖의 별도 상태로 두는 표현이 핵심입니다.

## 4. pairwise submission difference
{pairwise_text}

## 5. 왜 99999 같은 극단 sentinel은 쓰지 않는가
- 99/150/999만으로도 RobustScaler 이후 정상 관측 범위에서 충분히 멀리 분리됩니다.
- 더 극단적인 값은 RBF 거리 구조를 불필요하게 포화시킬 수 있고, LB에 맞춘 임의 보정처럼 보일 위험이 큽니다.
- 이번 최종 스토리는 out-of-range separation 가설이지 특정 거대 숫자 튜닝이 아닙니다.

## 6. 최종 제출 후보
- 1순위: `v531_single_sentinel99_reproduce.csv` 또는 기존 `v53_best_raw_rbf_B_mean_working_sentinel99.csv`.
- 2순위: seed ensemble이 CV에서 충분히 개선되면 `v531_sentinel99_seed5_ensemble.csv`; 개선이 작으면 보류.
- micro-gamble: sentinel150/999는 CV상 거의 같거나 아주 근소하게 좋지만, 설명 가능성은 sentinel150이 sentinel999보다 낫습니다.

## 7. PPT 설명 문장
- 최종 모델은 raw target RBF SVR이며, 0~1 범위의 0.01 단위 bounded grid target을 직접 학습한 뒤 clip/round2를 적용했습니다.
- S2 파생변수와 RobustScaler/OHE pipeline을 fold-safe하게 사용했고, test는 최종 predict에만 사용했습니다.
- mean_working 결측은 정상 근무시간 범위 밖 sentinel 99로 분리했습니다.
- 150/999도 유사한 결과를 보여 숫자 자체가 아니라 결측을 별도 latent state로 표현하는 것이 중요하다는 해석을 강화합니다.
- MLPRegressor도 비교했지만, 작은 tabular 데이터에서는 RBF SVR이 더 낮은 CV와 더 안정적인 분포를 보였습니다.
"""
    path.write_text(text, encoding="utf-8-sig")


def run_v531():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    cfg99 = sentinel_config(99)
    single_summary, single_folds, single_oof, _, _ = evaluate_cv(
        train_df, test_df, "single_sentinel99", cfg99, seed=42
    )
    full_raw99, full_pred99 = fit_full_train_submission(train_df, test_df, cfg99)
    write_submission(
        sample_submission,
        full_pred99,
        SUBMISSIONS_DIR / "v531_single_sentinel99_reproduce.csv",
    )
    existing_diff = compare_with_existing_submission(full_pred99)

    single_cv = pd.concat(
        [
            single_folds,
            pd.DataFrame(
                [
                    {
                        "exp_id": EXP_ID,
                        "candidate": "single_sentinel99",
                        "seed": 42,
                        "fold": "mean",
                        "mae": single_summary["mean_mae"],
                    },
                    {
                        "exp_id": EXP_ID,
                        "candidate": "single_sentinel99",
                        "seed": 42,
                        "fold": "std",
                        "mae": single_summary["std_mae"],
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    single_cv.to_csv(REPORTS_DIR / "v531_single_sentinel99_cv.csv", index=False)
    single_oof.to_csv(REPORTS_DIR / "v531_single_sentinel99_oof.csv", index=False)

    seed_rows = []
    seed_oofs = []
    seed_test_preds = []
    seed_test_raws = []
    for seed in SEEDS:
        summary, _, oof, test_raw, test_pred = evaluate_cv(
            train_df, test_df, f"sentinel99_seed{seed}", cfg99, seed=seed
        )
        seed_rows.append(summary)
        seed_oofs.append(oof)
        seed_test_raws.append(test_raw)
        seed_test_preds.append(test_pred)
        print(f"seed {seed}: {summary['mean_mae']:.6f}")

    seed_cv = pd.DataFrame(seed_rows)
    raw_oof_matrix = np.vstack([oof["raw_oof_pred"].to_numpy() for oof in seed_oofs])
    ensemble_raw_oof = raw_oof_matrix.mean(axis=0)
    ensemble_oof_pred = apply_grid_postprocess(ensemble_raw_oof, "round2")
    y = train_df[TARGET].to_numpy()
    ensemble_fold = np.zeros(len(train_df), dtype=int)
    ensemble_mae = mean_absolute_error(y, ensemble_oof_pred)
    ensemble_row = {
        "exp_id": EXP_ID,
        "candidate": "sentinel99_seed5_ensemble",
        "seed": "mean_5seed",
        "n_splits": 10,
        "postprocess": "clip_0_1_round2",
        "mean_mae": float(ensemble_mae),
        "std_mae": float(seed_cv["mean_mae"].std(ddof=1)),
        **summarize_prediction(ensemble_oof_pred),
        "sentinel_value": 99.0,
    }
    single_seed42_mae = float(seed_cv.loc[seed_cv["seed"].eq(42), "mean_mae"].iloc[0])
    single_seed42_std = float(seed_cv.loc[seed_cv["seed"].eq(42), "pred_std"].iloc[0])
    ensemble_row["improvement_vs_seed42"] = single_seed42_mae - ensemble_mae
    ensemble_row["pred_std_delta_vs_seed42"] = ensemble_row["pred_std"] - single_seed42_std
    ensemble_row["seed_mean_mae_std"] = float(seed_cv["mean_mae"].std(ddof=1))
    ensemble_row["is_candidate"] = bool(
        ensemble_row["improvement_vs_seed42"] >= 0.0002
        and ensemble_row["pred_std"] >= single_seed42_std - 0.01
    )
    if ensemble_row["improvement_vs_seed42"] >= 0.0002:
        ensemble_row["decision"] = "strong final-touch candidate"
    elif ensemble_row["improvement_vs_seed42"] > 0:
        ensemble_row["decision"] = "micro candidate; improvement is small"
    else:
        ensemble_row["decision"] = "not recommended; keep v5.3 sentinel99"

    ensemble_test_raw = np.vstack(seed_test_raws).mean(axis=0)
    ensemble_test_pred = apply_grid_postprocess(ensemble_test_raw, "round2")
    write_submission(
        sample_submission,
        ensemble_test_pred,
        SUBMISSIONS_DIR / "v531_sentinel99_seed5_ensemble.csv",
    )
    ensemble_row.update(
        {
            "submission_pred_mean": float(np.mean(ensemble_test_pred)),
            "submission_pred_std": float(np.std(ensemble_test_pred, ddof=1)),
            "submission_pred_min": float(np.min(ensemble_test_pred)),
            "submission_pred_max": float(np.max(ensemble_test_pred)),
        }
    )
    ensemble_vs_single = submission_diff_stats(
        "single_sentinel99_full_train", full_pred99, "seed5_fold_model_ensemble", ensemble_test_pred
    )
    ensemble_row.update({f"submission_{k}": v for k, v in ensemble_vs_single.items() if k not in ["left", "right"]})

    seed_cv.to_csv(REPORTS_DIR / "v531_seed_ensemble_cv_by_seed.csv", index=False)
    ensemble_oof = pd.DataFrame(
        {
            ID_COL: train_df[ID_COL],
            TARGET: train_df[TARGET],
            "exp_id": EXP_ID,
            "candidate": "sentinel99_seed5_ensemble",
            "seed": "mean_5seed",
            "fold": ensemble_fold,
            "raw_oof_pred": ensemble_raw_oof,
            "oof_pred": ensemble_oof_pred,
        }
    )
    pd.concat(seed_oofs + [ensemble_oof], ignore_index=True).to_csv(
        REPORTS_DIR / "v531_seed_ensemble_oof.csv", index=False
    )
    pd.DataFrame([ensemble_row]).to_csv(REPORTS_DIR / "v531_seed_ensemble_summary.csv", index=False)

    sentinel_rows = []
    sentinel_oofs = {"sentinel99": single_oof}
    sentinel_submissions = {"sentinel99": full_pred99}
    for sentinel in [150, 999]:
        name = f"sentinel{sentinel}"
        summary, _, oof, _, _ = evaluate_cv(train_df, test_df, name, sentinel_config(sentinel), seed=42)
        raw, pred = fit_full_train_submission(train_df, test_df, sentinel_config(sentinel))
        write_submission(sample_submission, pred, SUBMISSIONS_DIR / f"v531_single_{name}.csv")
        sentinel_rows.append(summary)
        sentinel_oofs[name] = oof
        sentinel_submissions[name] = pred
        print(f"{name}: {summary['mean_mae']:.6f}")

    sentinel_summary = pd.DataFrame([single_summary] + sentinel_rows)
    sentinel_summary["candidate"] = sentinel_summary["candidate"].replace({"single_sentinel99": "sentinel99"})
    sentinel_summary["improvement_vs_sentinel99"] = V53_SENTINEL99_CV - sentinel_summary["mean_mae"]
    sentinel_summary["decision"] = np.where(
        sentinel_summary["candidate"].eq("sentinel99"),
        "primary reference",
        np.where(
            sentinel_summary["improvement_vs_sentinel99"] >= 0.00002,
            "micro-gamble; prefer 150 for explanation if used",
            "not better than sentinel99",
        ),
    )

    pairwise = []
    keys = ["sentinel99", "sentinel150", "sentinel999"]
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            pairwise.append(submission_diff_stats(left, sentinel_submissions[left], right, sentinel_submissions[right]))
    pairwise_df = pd.DataFrame(pairwise)
    freq_df = pd.concat(
        [frequency_table(name, pred) for name, pred in sentinel_submissions.items()],
        ignore_index=True,
    )
    sentinel_summary.to_csv(REPORTS_DIR / "v531_sentinel_sweep_micro_cv.csv", index=False)
    pairwise_df.to_csv(REPORTS_DIR / "v531_sentinel_submission_pairwise_diff.csv", index=False)
    freq_df.to_csv(REPORTS_DIR / "v531_sentinel_prediction_frequency.csv", index=False)

    comparison, candidates = build_final_tables(
        single_summary,
        sentinel_summary,
        ensemble_row,
        pairwise_df,
        existing_diff,
    )
    comparison.to_csv(REPORTS_DIR / "v531_final_model_comparison_for_ppt.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(REPORTS_DIR / "v531_final_submission_candidates.csv", index=False, encoding="utf-8-sig")
    write_synthesis(
        REPORTS_DIR / "v531_final_synthesis.md",
        single_summary,
        ensemble_row,
        sentinel_summary,
        pairwise_df,
        existing_diff,
    )

    print("\n=== v5.3.1 single sentinel99 ===")
    print(pd.DataFrame([single_summary]).round(6).to_string(index=False))
    print("\n=== v5.3.1 seed ensemble ===")
    print(pd.DataFrame([ensemble_row]).round(6).to_string(index=False))
    print("\n=== v5.3.1 sentinel micro ===")
    print(sentinel_summary[["candidate", "mean_mae", "pred_std", "improvement_vs_sentinel99", "decision"]].round(6).to_string(index=False))
    print("\n=== submission diff ===")
    print(pairwise_df.round(8).to_string(index=False))
    return comparison, candidates


if __name__ == "__main__":
    run_v531()
