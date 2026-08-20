import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from src.config import ID_COL, N_SPLITS, N_TARGET_BINS, RANDOM_STATE, TARGET
from src.postprocess import POSTPROCESSORS


def make_stratified_regression_bins(y, n_bins=N_TARGET_BINS):
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def make_cv(y, n_splits=N_SPLITS):
    bins = make_stratified_regression_bins(y)
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    ).split(np.zeros(len(y)), bins)


def evaluate_pipeline(name, pipeline, train_df):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()

    fold_rows = []
    oof_by_postprocess = {post_name: np.zeros(len(train_df), dtype=float) for post_name in POSTPROCESSORS}

    for fold, (tr_idx, va_idx) in enumerate(make_cv(y), start=1):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_valid = y[tr_idx], y[va_idx]

        fold_model = clone(pipeline)
        fold_model.fit(X_train, y_train)
        valid_pred = fold_model.predict(X_valid)

        for post_name, post_func in POSTPROCESSORS.items():
            processed_pred = post_func(valid_pred)
            oof_by_postprocess[post_name][va_idx] = processed_pred
            mae = mean_absolute_error(y_valid, processed_pred)
            fold_rows.append(
                {
                    "model": name,
                    "postprocess": post_name,
                    "fold": fold,
                    "mae": mae,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = (
        fold_df.groupby(["model", "postprocess"], as_index=False)
        .agg(mean_mae=("mae", "mean"), std_mae=("mae", "std"))
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    return fold_df, summary_df


def evaluate_pipeline_with_oof(name, pipeline, train_df):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()

    fold_rows = []
    oof_rows = []

    for fold, (tr_idx, va_idx) in enumerate(make_cv(y), start=1):
        X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
        y_train, y_valid = y[tr_idx], y[va_idx]

        fold_model = clone(pipeline)
        fold_model.fit(X_train, y_train)
        valid_pred = fold_model.predict(X_valid)

        for post_name, post_func in POSTPROCESSORS.items():
            processed_pred = post_func(valid_pred)
            mae = mean_absolute_error(y_valid, processed_pred)
            fold_rows.append(
                {
                    "model": name,
                    "postprocess": post_name,
                    "fold": fold,
                    "mae": mae,
                }
            )

            oof_rows.extend(
                {
                    ID_COL: row_id,
                    TARGET: target,
                    "model": name,
                    "postprocess": post_name,
                    "oof_pred": pred,
                    "fold": fold,
                }
                for row_id, target, pred in zip(
                    X_valid[ID_COL].to_numpy(),
                    y_valid,
                    processed_pred,
                )
            )

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    pred_stats = (
        oof_df.groupby(["model", "postprocess"], as_index=False)["oof_pred"]
        .agg(pred_std_oof="std", pred_min_oof="min", pred_max_oof="max")
    )
    summary_df = (
        fold_df.groupby(["model", "postprocess"], as_index=False)
        .agg(mean_mae=("mae", "mean"), std_mae=("mae", "std"))
        .merge(pred_stats, on=["model", "postprocess"], how="left")
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    return fold_df, summary_df, oof_df
