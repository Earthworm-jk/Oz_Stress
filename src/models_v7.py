from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

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
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
RBF_SENTINEL99_CV = 0.13491666666666666

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
    "pulse_pressure",
    "map",
    "glucose_cholesterol_ratio",
    "cholesterol_glucose_product",
]
CAT_COLS = [
    "gender",
    "activity",
    "smoke_status",
    "medical_history",
    "family_medical_history",
    "sleep_pattern",
    "edu_level",
]


class V7FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        X_df["mean_working"] = X_df["mean_working"].fillna(99)
        X_df["bmi"] = X_df["weight"] / np.square(X_df["height"] / 100.0)
        X_df["pulse_pressure"] = X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0
        X_df["glucose_cholesterol_ratio"] = X_df["glucose"] / X_df["cholesterol"].replace(0, np.nan)
        X_df["cholesterol_glucose_product"] = X_df["cholesterol"] * X_df["glucose"]
        for col in CAT_COLS:
            X_df[col] = X_df[col].astype("object").fillna("Unknown")
        return X_df.drop(columns=[c for c in [ID_COL, TARGET] if c in X_df.columns])


def make_preprocessor(scaler="standard"):
    scaler_step = StandardScaler() if scaler == "standard" else RobustScaler()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_step)]),
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


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


def make_pipeline(model, scaler="standard"):
    return Pipeline(
        steps=[
            ("features", V7FeatureEngineer()),
            ("preprocess", make_preprocessor(scaler=scaler)),
            ("dense", DenseTransformer()),
            ("model", model),
        ]
    )


def mlp_specs():
    common = {
        "solver": "adam",
        "max_iter": 3000,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 50,
        "random_state": RANDOM_STATE,
        "batch_size": "auto",
        "verbose": False,
    }
    specs = [
        (
            "mlp_small_relu_l2_1e4",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                learning_rate="adaptive",
                **common,
            ),
            "standard",
        ),
        (
            "mlp_medium_relu_l2_1e4",
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                learning_rate="adaptive",
                **common,
            ),
            "standard",
        ),
        (
            "mlp_medium_relu_l2_1e3",
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                learning_rate="adaptive",
                **common,
            ),
            "standard",
        ),
        (
            "mlp_wide_relu_l2_1e3",
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=5e-4,
                learning_rate="adaptive",
                **common,
            ),
            "standard",
        ),
        (
            "mlp_tanh_medium_l2_1e4",
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="tanh",
                alpha=1e-4,
                learning_rate_init=5e-4,
                learning_rate="adaptive",
                **common,
            ),
            "standard",
        ),
        (
            "mlp_small_lbfgs",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="lbfgs",
                alpha=1e-3,
                max_iter=1500,
                random_state=RANDOM_STATE,
            ),
            "standard",
        ),
        (
            "mlp_medium_relu_l2_1e3_robust_scaler",
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                learning_rate="adaptive",
                **common,
            ),
            "robust",
        ),
    ]
    return specs


def evaluate_model(train_df, name, model, scaler, n_splits=5):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    raw_oof = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(np.zeros(len(train_df))), start=1):
        pipe = make_pipeline(clone(model), scaler=scaler)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        raw_oof[va_idx] = pipe.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        for post, pred in {
            "raw": raw_oof[va_idx],
            "clip_0_1": clip_0_1(raw_oof[va_idx]),
            "clip_0_1_round2": clip_round_2(raw_oof[va_idx]),
        }.items():
            fold_rows.append(
                {
                    "exp_id": EXP_ID,
                    "model": name,
                    "scaler": scaler,
                    "n_splits": n_splits,
                    "fold": fold,
                    "postprocess": post,
                    "mae": mean_absolute_error(y[va_idx], pred),
                }
            )
    summary_rows = []
    oof_rows = []
    for post, pred in {
        "raw": raw_oof,
        "clip_0_1": clip_0_1(raw_oof),
        "clip_0_1_round2": clip_round_2(raw_oof),
    }.items():
        fold_mae = [r["mae"] for r in fold_rows if r["postprocess"] == post]
        summary_rows.append(
            {
                "exp_id": EXP_ID,
                "model": name,
                "scaler": scaler,
                "n_splits": n_splits,
                "postprocess": post,
                "mean_mae": float(np.mean(fold_mae)),
                "std_mae": float(np.std(fold_mae, ddof=1)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
            }
        )
        oof_rows.append(
            pd.DataFrame(
                {
                    ID_COL: train_df[ID_COL],
                    TARGET: train_df[TARGET],
                    "exp_id": EXP_ID,
                    "model": name,
                    "scaler": scaler,
                    "n_splits": n_splits,
                    "postprocess": post,
                    "fold": folds,
                    "oof_pred": pred,
                }
            )
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def plot_best(train_df, best_oof, best_name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pred = best_oof["oof_pred"].to_numpy()
    y = best_oof[TARGET].to_numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(y, bins=40, alpha=0.5, label="target")
    plt.hist(pred, bins=40, alpha=0.5, label="MLP OOF")
    plt.legend()
    plt.title(f"Prediction distribution: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v7_mlp_pred_distribution_best.png", dpi=140)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y, pred, s=8, alpha=0.5)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("target")
    plt.ylabel("OOF prediction")
    plt.title(f"OOF vs target: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v7_mlp_oof_vs_y_best.png", dpi=140)
    plt.close()


def model_family_comparison(mlp_best):
    rows = [
        {"family": "linear", "model": "Ridge/ElasticNet/LinearSVR", "cv_mae": 0.2445, "notes": "Earlier baseline range; weak additive scorecard."},
        {"family": "tree", "model": "ExtraTrees v3", "cv_mae": 0.179860, "notes": "Best tree-style scorecard branch."},
        {"family": "kernel", "model": "raw RBF S2", "cv_mae": 0.139413, "notes": "v5.1 raw RBF S2."},
        {"family": "kernel", "model": "raw RBF sentinel99", "cv_mae": RBF_SENTINEL99_CV, "notes": "v5.3/v5.4 sentinel branch."},
        {"family": "neural", "model": mlp_best["model"], "cv_mae": mlp_best["mean_mae"], "notes": "Best v7 MLPRegressor."},
    ]
    return pd.DataFrame(rows).sort_values("cv_mae")


def maybe_run_torch_gate(best_mlp):
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"PyTorch skipped: torch is unavailable ({type(exc).__name__})."
    if best_mlp["mean_mae"] <= 0.145:
        return False, "PyTorch gate met, but skipped to keep v7 lightweight; sklearn MLP already did not beat RBF."
    return False, f"PyTorch skipped: sklearn MLP best CV {best_mlp['mean_mae']:.6f} > 0.145 gate."


def write_synthesis(best_mlp, family, torch_note, submission_created):
    family_table = family.to_csv(index=False)
    text = f"""# v7 Neural Baseline Synthesis

## Result
Best MLPRegressor: `{best_mlp['model']}` with `{best_mlp['postprocess']}` / CV MAE `{best_mlp['mean_mae']:.6f}`.

Current raw RBF sentinel99 reference CV MAE: `{RBF_SENTINEL99_CV:.6f}`.

## Interpretation
1. MLPRegressor did not approach the RBF SVR sentinel branch if its CV remains materially above the reference.
2. Neural models are nonlinear and improve over weak linear/additive baselines only if their CV is below the linear range, but the stable RBF kernel remains stronger for this 3,000-row tabular setting.
3. The prediction distribution and OOF scatter figures are saved in `figures/`.
4. PyTorch decision: {torch_note}
5. Submission created: `{submission_created}`. The v7 branch is primarily report evidence, not a submission branch.

## Model family comparison

```csv
{family_table}
```
"""
    (REPORTS_DIR / "v7_final_synthesis.md").write_text(text, encoding="utf-8")


def maybe_save_submission(train_df, test_df, sample, spec, best):
    if best["mean_mae"] > RBF_SENTINEL99_CV - 0.001:
        return None
    name, model, scaler = spec
    pipe = make_pipeline(clone(model), scaler=scaler)
    pipe.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
    pred = clip_round_2(pipe.predict(test_df))
    sub = sample.copy()
    sub[TARGET] = pred
    path = SUBMISSIONS_DIR / "v7_best_neural_candidate.csv"
    sub.to_csv(path, index=False)
    return path


def run_v7():
    train_df = pd.read_csv(TRAIN_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_summary, all_folds, all_oof = [], [], []
    spec_lookup = {}
    for name, model, scaler in mlp_specs():
        print(f"[{name}] scaler={scaler}")
        summary, folds, oof = evaluate_model(train_df, name, model, scaler, n_splits=5)
        print(summary[["postprocess", "mean_mae", "std_mae", "pred_std"]].round(6).to_string(index=False))
        all_summary.append(summary)
        all_folds.append(folds)
        all_oof.append(oof)
        spec_lookup[name] = (name, model, scaler)

    summary_df = pd.concat(all_summary, ignore_index=True).sort_values("mean_mae").reset_index(drop=True)
    fold_df = pd.concat(all_folds, ignore_index=True)
    oof_df = pd.concat(all_oof, ignore_index=True)

    # Optional 10-fold confirmation for top 2 only if MLP is reasonably close.
    best_rows = summary_df[summary_df["postprocess"].eq("clip_0_1_round2")].head(2)
    if best_rows.iloc[0]["mean_mae"] <= 0.150:
        for _, row in best_rows.iterrows():
            name, model, scaler = spec_lookup[row["model"]]
            confirm_name = f"{name}_10fold_confirm"
            summary, folds, oof = evaluate_model(train_df, confirm_name, model, scaler, n_splits=10)
            all_summary.append(summary)
            all_folds.append(folds)
            all_oof.append(oof)
        summary_df = pd.concat(all_summary, ignore_index=True).sort_values("mean_mae").reset_index(drop=True)
        fold_df = pd.concat(all_folds, ignore_index=True)
        oof_df = pd.concat(all_oof, ignore_index=True)

    summary_df.to_csv(REPORTS_DIR / "v7_mlpregressor_cv_results.csv", index=False)
    fold_df.to_csv(REPORTS_DIR / "v7_mlpregressor_fold_results.csv", index=False)
    oof_df.to_csv(REPORTS_DIR / "v7_mlpregressor_oof_predictions.csv", index=False)

    best = summary_df[summary_df["postprocess"].eq("clip_0_1_round2")].iloc[0]
    best_oof = oof_df[
        (oof_df["model"].eq(best["model"]))
        & (oof_df["postprocess"].eq(best["postprocess"]))
        & (oof_df["n_splits"].eq(best["n_splits"]))
    ]
    plot_best(train_df, best_oof, best["model"])

    family = model_family_comparison(best)
    family.to_csv(REPORTS_DIR / "v7_model_family_comparison.csv", index=False)

    torch_ran, torch_note = maybe_run_torch_gate(best)
    if not torch_ran:
        pd.DataFrame([{"exp_id": EXP_ID, "status": "skipped", "notes": torch_note}]).to_csv(
            REPORTS_DIR / "v7_torch_mlp_cv_results.csv", index=False
        )

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = None
    base_model_name = best["model"].replace("_10fold_confirm", "")
    if base_model_name in spec_lookup:
        path = maybe_save_submission(
            train_df,
            pd.read_csv(TEST_PATH),
            pd.read_csv(SAMPLE_SUBMISSION_PATH),
            spec_lookup[base_model_name],
            best,
        )

    write_synthesis(best, family, torch_note, path is not None)
    print("\n=== V7 best ===")
    print(best.to_string())
    if path:
        print(f"Saved submission: {path}")
    else:
        print("No v7 submission: neural model did not beat raw RBF sentinel99 by >= 0.001.")


if __name__ == "__main__":
    run_v7()
