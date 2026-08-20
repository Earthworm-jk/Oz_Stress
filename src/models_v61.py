from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, _tree

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
from src.models_v5 import DenseTransformer, TargetModeRegressor
from src.models_v54 import V54FeatureEngineer, feature_columns
from src.postprocess import clip_0_1, clip_round_2


EXP_ID = f"v61_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"
BASE_C = 3.963530707518144
BASE_GAMMA = 1.0631617004546035
BASELINE_CV = 0.13491666666666666
SPLITTER = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)


def make_config():
    return {"mean_working_mode": "sentinel", "sentinel_value": 99.0}


def make_preprocessor():
    config = make_config()
    numeric, ohe = feature_columns(config)
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]),
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
                ohe,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_rbf_pipeline(c=BASE_C, gamma=BASE_GAMMA, epsilon=0.0):
    return Pipeline(
        steps=[
            ("features", V54FeatureEngineer(**make_config())),
            ("preprocess", make_preprocessor()),
            ("dense", DenseTransformer()),
            (
                "model",
                TargetModeRegressor(
                    SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon, shrinking=True, cache_size=500),
                    target_mode="raw",
                ),
            ),
        ]
    )


def evaluate_pipeline(train_df, pipeline, name):
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].to_numpy()
    raw = np.zeros(len(train_df), dtype=float)
    folds = np.zeros(len(train_df), dtype=int)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(SPLITTER.split(np.zeros(len(y))), start=1):
        model = clone(pipeline)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        raw[va_idx] = model.predict(X.iloc[va_idx])
        folds[va_idx] = fold
        for post, pred in {"clip": clip_0_1(raw[va_idx]), "round2": clip_round_2(raw[va_idx])}.items():
            fold_rows.append({"candidate": name, "fold": fold, "postprocess": post, "mae": mean_absolute_error(y[va_idx], pred)})
    rows = []
    oof_parts = []
    for post, pred in {"clip": clip_0_1(raw), "round2": clip_round_2(raw)}.items():
        mae_by_fold = [r["mae"] for r in fold_rows if r["postprocess"] == post]
        rows.append(
            {
                "exp_id": EXP_ID,
                "candidate": name,
                "postprocess": post,
                "mean_mae": float(np.mean(mae_by_fold)),
                "std_mae": float(np.std(mae_by_fold, ddof=1)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred, ddof=1)),
                "pred_min": float(np.min(pred)),
                "pred_max": float(np.max(pred)),
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    ID_COL: train_df[ID_COL],
                    TARGET: train_df[TARGET],
                    "exp_id": EXP_ID,
                    "candidate": name,
                    "postprocess": post,
                    "fold": folds,
                    "raw_oof_pred": raw,
                    "oof_pred": pred,
                }
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True)


def rbf_local_tuning(train_df):
    rows, fold_rows, oofs = [], [], []
    first_pass = []
    for cm in [0.7, 0.85, 1.0, 1.15, 1.3]:
        for gm in [0.7, 0.85, 1.0, 1.15, 1.3]:
            c, gamma = BASE_C * cm, BASE_GAMMA * gm
            name = f"c{cm:g}_g{gm:g}_e0"
            res, folds, oof = evaluate_pipeline(train_df, make_rbf_pipeline(c, gamma, 0.0), name)
            res["C"], res["gamma"], res["epsilon"] = c, gamma, 0.0
            rows.append(res)
            fold_rows.append(folds.assign(C=c, gamma=gamma, epsilon=0.0))
            oofs.append(oof)
            first_pass.append((name, c, gamma, float(res[res["postprocess"].eq("round2")]["mean_mae"].iloc[0])))
            print(f"tune {name}: {first_pass[-1][3]:.6f}")
    top5 = sorted(first_pass, key=lambda x: x[3])[:5]
    for base_name, c, gamma, _ in top5:
        for eps in [0.001, 0.002, 0.005]:
            name = f"{base_name}_eps{eps:g}"
            res, folds, oof = evaluate_pipeline(train_df, make_rbf_pipeline(c, gamma, eps), name)
            res["C"], res["gamma"], res["epsilon"] = c, gamma, eps
            rows.append(res)
            fold_rows.append(folds.assign(C=c, gamma=gamma, epsilon=eps))
            oofs.append(oof)
            print(f"tune {name}: {res[res['postprocess'].eq('round2')]['mean_mae'].iloc[0]:.6f}")
    return pd.concat(rows, ignore_index=True).sort_values("mean_mae"), pd.concat(fold_rows, ignore_index=True), pd.concat(oofs, ignore_index=True)


def target_lattice(train_df):
    y100 = (train_df[TARGET] * 100).round().astype(int)
    counts = y100.value_counts().sort_index()
    entropy = -np.sum((counts / counts.sum()) * np.log2(counts / counts.sum()))
    rows = [
        {"metric": "is_integer_grid", "value": bool(np.allclose(train_df[TARGET] * 100, y100))},
        {"metric": "unique_classes", "value": int(y100.nunique())},
        {"metric": "class_count_mean", "value": float(counts.mean())},
        {"metric": "class_count_std", "value": float(counts.std())},
        {"metric": "class_count_min", "value": int(counts.min())},
        {"metric": "class_count_max", "value": int(counts.max())},
        {"metric": "entropy_bits", "value": float(entropy)},
    ]
    for cls in [0, 1, 2, 3, 97, 98, 99, 100]:
        rows.append({"metric": f"count_y100_{cls}", "value": int(counts.get(cls, 0))})
    for mod in [2, 5, 10]:
        freq = y100.mod(mod).value_counts(normalize=True).sort_index()
        for k, v in freq.items():
            rows.append({"metric": f"mod_{mod}_rem_{k}_rate", "value": float(v)})
    rows.append({"metric": "interpretation", "value": "target is exactly a 0..100 integer lattice, supporting bounded synthetic/grid score hypothesis"})

    plt.figure(figsize=(12, 4))
    counts.reindex(range(101), fill_value=0).plot(kind="bar", width=1.0)
    plt.title("y100 class frequency")
    plt.xlabel("y100 class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v61_y100_class_frequency.png", dpi=140)
    plt.close()
    return pd.DataFrame(rows), counts.rename_axis("y100").reset_index(name="count")


def compact_feature_frame(train_df):
    df = train_df.copy()
    df["mean_working_sentinel99"] = df["mean_working"].fillna(99)
    df["BMI"] = df["weight"] / np.square(df["height"] / 100)
    maps = {
        "gender": {"F": 0, "M": 1},
        "activity": {"light": 0, "moderate": 1, "intense": 2},
        "sleep_pattern": {"sleep difficulty": 0, "normal": 1, "oversleeping": 2},
        "edu_level": {"high school diploma": 1, "bachelors degree": 2, "graduate degree": 3},
    }
    for col, mp in maps.items():
        df[col + "_enc"] = df[col].map(mp).fillna(0)
    for col in ["smoke_status", "medical_history", "family_medical_history"]:
        codes = {v: i for i, v in enumerate(sorted(df[col].astype("object").fillna("Unknown").unique()))}
        df[col + "_enc"] = df[col].astype("object").fillna("Unknown").map(codes)
    cols = [
        "age", "BMI", "cholesterol", "glucose", "systolic_blood_pressure",
        "diastolic_blood_pressure", "bone_density", "mean_working_sentinel99",
        "sleep_pattern_enc", "activity_enc", "medical_history_enc", "family_medical_history_enc",
    ]
    return df[cols].fillna(df[cols].median()), cols


def symbolic_approximation(train_df, base_oof):
    X, cols = compact_feature_frame(train_df)
    targets = {
        "y100": train_df[TARGET].to_numpy() * 100,
        "pred100": base_oof.sort_values(ID_COL)["oof_pred"].to_numpy() * 100,
    }
    rows, terms = [], []
    for target_name, y in targets.items():
        for degree in [1, 2]:
            pipe = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                    ("scale", RobustScaler()),
                    ("model", ElasticNet(alpha=0.003, l1_ratio=0.2, max_iter=20000, random_state=RANDOM_STATE)),
                ]
            )
            pipe.fit(X, y)
            pred = pipe.predict(X)
            names = pipe.named_steps["poly"].get_feature_names_out(cols)
            coefs = pipe.named_steps["model"].coef_
            complexity = int(np.sum(np.abs(coefs) > 1e-8))
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "target": target_name,
                    "model": f"elasticnet_poly_degree_{degree}",
                    "mae": float(mean_absolute_error(y, pred)),
                    "r2": float(r2_score(y, pred)),
                    "formula_complexity": complexity,
                    "notes": "Sparse formula-like fallback; symbolic packages not installed/used.",
                }
            )
            top_idx = np.argsort(np.abs(coefs))[::-1][:15]
            for idx in top_idx:
                terms.append(
                    {
                        "exp_id": EXP_ID,
                        "target": target_name,
                        "model": f"elasticnet_poly_degree_{degree}",
                        "term": names[idx],
                        "coefficient": float(coefs[idx]),
                        "abs_coefficient": float(abs(coefs[idx])),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(terms)


def counterfactual_probing(train_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = make_rbf_pipeline()
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
    picks = []
    train_df = train_df.copy()
    train_df["decile"] = pd.qcut(train_df[TARGET], q=10, labels=False, duplicates="drop")
    for d in sorted(train_df["decile"].dropna().unique()):
        picks.extend(train_df[train_df["decile"].eq(d)].head(3).index.tolist())
    picks.extend(train_df[train_df["mean_working"].isna()].head(10).index.tolist())
    picks.extend(train_df.sort_values(TARGET).head(10).index.tolist())
    picks.extend(train_df.sort_values(TARGET, ascending=False).head(10).index.tolist())
    sample = train_df.loc[sorted(set(picks))].drop(columns=["decile"])
    base_pred = model.predict(sample.drop(columns=[TARGET]))
    rows = []
    probes = {
        "mean_working": [0, 5, 6, 7, 8, 9, 10, 11, 12, 15, 99, 150],
        "sleep_pattern": ["normal", "oversleeping", "sleep difficulty"],
        "activity": ["light", "moderate", "intense"],
        "smoke_status": sorted(train_df["smoke_status"].dropna().unique()),
        "glucose": list(train_df["glucose"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])),
        "cholesterol": list(train_df["cholesterol"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])),
        "systolic_blood_pressure": list(train_df["systolic_blood_pressure"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])),
        "diastolic_blood_pressure": list(train_df["diastolic_blood_pressure"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])),
        "bone_density": list(train_df["bone_density"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])),
    }
    for var, values in probes.items():
        for value in values:
            tmp = sample.copy()
            tmp[var] = value
            pred = model.predict(tmp.drop(columns=[TARGET]))
            rows.append({"variable": var, "value": str(value), "delta_mean": float(np.mean(pred - base_pred)), "pred_mean": float(np.mean(pred)), "pred_std": float(np.std(pred, ddof=1))})
    cf = pd.DataFrame(rows)
    for group, vars_ in {
        "mean_working": ["mean_working"],
        "sleep_activity": ["sleep_pattern", "activity"],
        "metabolic_bp": ["glucose", "cholesterol", "systolic_blood_pressure", "diastolic_blood_pressure", "bone_density"],
    }.items():
        plt.figure(figsize=(8, 4))
        for var in vars_:
            sub = cf[cf["variable"].eq(var)]
            plt.plot(range(len(sub)), sub["delta_mean"], marker="o", label=var)
        plt.legend()
        plt.title(f"Counterfactual delta: {group}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"v61_counterfactual_{group}.png", dpi=140)
        plt.close()
    return cf


def sample_geometry(train_df, base_oof):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pre = Pipeline([("features", V54FeatureEngineer(**make_config())), ("preprocess", make_preprocessor()), ("dense", DenseTransformer())])
    X = pre.fit_transform(train_df.drop(columns=[TARGET]))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    df = train_df[[ID_COL, TARGET, "mean_working"]].copy()
    df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]
    df = df.merge(base_oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    df["residual"] = df[TARGET] - df["oof_pred"]
    df["mean_working_missing"] = df["mean_working"].isna()
    for color_col, name in [(TARGET, "stress_score"), ("mean_working_missing", "mean_working_missing"), ("residual", "residual")]:
        plt.figure(figsize=(6, 5))
        plt.scatter(df["pc1"], df["pc2"], c=df[color_col].astype(float), s=8, cmap="viridis")
        plt.colorbar()
        plt.title(f"PCA colored by {name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"v61_pca_{name}.png", dpi=140)
        plt.close()
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(np.arange(len(df)), size=min(800, len(df)), replace=False)
    Xs = X[idx]
    ys = train_df[TARGET].to_numpy()[idx]
    dists = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    ydiff = np.abs(ys[:, None] - ys[None, :])
    tri = np.triu_indices(len(idx), k=1)
    corr = np.corrcoef(dists[tri], ydiff[tri])[0, 1]
    missing = df["mean_working_missing"].to_numpy()
    center_missing = X[missing].mean(axis=0)
    center_observed = X[~missing].mean(axis=0)
    return pd.DataFrame(
        [
            {
                "exp_id": EXP_ID,
                "pca_explained_var_1": float(pca.explained_variance_ratio_[0]),
                "pca_explained_var_2": float(pca.explained_variance_ratio_[1]),
                "feature_distance_y_absdiff_corr": float(corr),
                "missing_vs_observed_center_distance": float(np.linalg.norm(center_missing - center_observed)),
                "notes": "Train-only geometry diagnostic on sentinel99 preprocessing.",
            }
        ]
    )


def rbf_distillation(train_df, base_oof):
    X = train_df.drop(columns=[TARGET])
    y = base_oof.set_index(ID_COL).loc[train_df[ID_COL], "oof_pred"].to_numpy() * 100
    models = [
        ("extra_trees", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1)),
        ("tree_depth3", DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=RANDOM_STATE)),
        ("tree_depth5", DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE)),
        ("hist_gb", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, random_state=RANDOM_STATE)),
    ]
    rows, rules, importances = [], [], []
    for name, estimator in models:
        pipe = Pipeline([("features", V54FeatureEngineer(**make_config())), ("preprocess", make_preprocessor()), ("dense", DenseTransformer()), ("model", estimator)])
        pred = np.zeros(len(train_df), dtype=float)
        for tr_idx, va_idx in SPLITTER.split(np.zeros(len(train_df))):
            model = clone(pipe)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            pred[va_idx] = model.predict(X.iloc[va_idx])
        rows.append({"exp_id": EXP_ID, "model": name, "mae_to_rbf_pred100": float(mean_absolute_error(y, pred)), "correlation": float(np.corrcoef(y, pred)[0, 1]), "r2": float(r2_score(y, pred))})
        pipe.fit(X, y)
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            feats = pipe.named_steps["preprocess"].get_feature_names_out()
            for f, imp in zip(feats, pipe.named_steps["model"].feature_importances_):
                importances.append({"exp_id": EXP_ID, "model": name, "feature": f, "importance": float(imp)})
        if isinstance(estimator, DecisionTreeRegressor):
            feats = pipe.named_steps["preprocess"].get_feature_names_out()
            for node, rule in tree_rules(pipe.named_steps["model"], feats).items():
                rules.append({"exp_id": EXP_ID, "model": name, "rule": rule})
    return pd.DataFrame(rows), pd.DataFrame(rules), pd.DataFrame(importances).sort_values("importance", ascending=False)


def tree_rules(tree, names):
    t = tree.tree_
    rules = {}

    def rec(node, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = " AND ".join(path) if path else "ALL"
            return
        name = names[t.feature[node]]
        th = t.threshold[node]
        rec(t.children_left[node], path + [f"{name} <= {th:.4f}"])
        rec(t.children_right[node], path + [f"{name} > {th:.4f}"])

    rec(0, [])
    return rules


def residual_topology(train_df, base_oof):
    df = train_df.merge(base_oof[[ID_COL, "oof_pred"]], on=ID_COL, how="left")
    df["residual100"] = (df[TARGET] - df["oof_pred"]) * 100
    df["group"] = np.select([df["residual100"] >= 15, df["residual100"] <= -15, df["residual100"].abs() <= 5], ["large_positive", "large_negative", "normal"], default="middle")
    profile = df.groupby("group", as_index=False).agg(count=(TARGET, "size"), y_mean=(TARGET, "mean"), pred_mean=("oof_pred", "mean"), residual_mean=("residual100", "mean"), mean_working_missing_rate=("mean_working", lambda s: s.isna().mean()))
    pre = Pipeline([("features", V54FeatureEngineer(**make_config())), ("preprocess", make_preprocessor()), ("dense", DenseTransformer())])
    X = pre.fit_transform(train_df.drop(columns=[TARGET]))
    rows = []
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        tmp = df.copy()
        tmp["cluster"] = labels
        cl = tmp.groupby("cluster", as_index=False).agg(count=(TARGET, "size"), y_mean=(TARGET, "mean"), pred_mean=("oof_pred", "mean"), residual_mean=("residual100", "mean"))
        cl.insert(0, "k", k)
        rows.append(cl)
    return pd.concat(rows, ignore_index=True), profile


def final_synthesis(best_tuning, lattice, symbolic, geometry, distill, residual_clusters, submission_created):
    text = f"""# v6.1 Final Synthesis

## 1. RBF local tuning
Best local-tuning row: `{best_tuning['candidate']}` / `{best_tuning['postprocess']}` with CV MAE `{best_tuning['mean_mae']:.6f}`.
Baseline sentinel99 CV MAE is approximately `{BASELINE_CV:.6f}`. A submission candidate is created only if improvement is at least 0.0005.

## 2. Target lattice
The target is an exact 0..100 integer lattice after multiplying by 100. This strengthens the bounded synthetic/grid score hypothesis.

## 3. Symbolic approximation
Sparse polynomial approximations are diagnostic only. If pred100 is easier to approximate than y100, it suggests the RBF is smoothing a latent score surface rather than reconstructing a simple explicit formula.

## 4. Counterfactual probing
Counterfactual reports in `v61_counterfactual_probing.csv` show how the fitted raw RBF reacts to mean_working sentinel values, sleep/activity categories, metabolic variables, blood pressure, and bone density.

## 5. Sample geometry
PCA and distance diagnostics are saved in `v61_sample_geometry_summary.csv` and figures. Missing mean_working geometry is evaluated without looking at test distribution.

## 6. RBF distillation
Distillation fidelity is summarized in `v61_rbf_distillation_fidelity.csv`. This indicates how much of the RBF prediction function can be compressed into explainable tree-like structure.

## 7. Residual topology
Residual clusters and large residual profiles are saved in `v61_residual_topology_clusters.csv` and `v61_large_residual_profile.csv`. These are diagnostic and not direct correction rules.

## 8. Submission decision
Submission created: `{submission_created}`.
If no submission was created, the safer final candidate remains the v5/v54 raw RBF sentinel candidate.
"""
    (REPORTS_DIR / "v61_final_synthesis.md").write_text(text, encoding="utf-8")


def maybe_save_submission(train_df, test_df, sample, best_row):
    if best_row["mean_mae"] > BASELINE_CV - 0.0005 or best_row["postprocess"] != "round2":
        return None
    # parse candidate name: c{cm}_g{gm}_e0...
    row = best_row
    c = row["C"]
    gamma = row["gamma"]
    eps = row["epsilon"]
    model = make_rbf_pipeline(c, gamma, eps)
    model.fit(train_df.drop(columns=[TARGET]), train_df[TARGET])
    pred = clip_round_2(model.predict(test_df))
    sub = sample.copy()
    sub[TARGET] = pred
    path = SUBMISSIONS_DIR / "v61_best_raw_rbf_sentinel99_tuned.csv"
    sub.to_csv(path, index=False)
    return path


def run_v61_experiments():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)

    tuning, tuning_folds, tuning_oof = rbf_local_tuning(train_df)
    tuning.to_csv(REPORTS_DIR / "v61_rbf_local_tuning.csv", index=False)
    tuning_oof.to_csv(REPORTS_DIR / "oof_predictions_v61_rbf_tuning.csv", index=False)
    tuning_folds.to_csv(REPORTS_DIR / "v61_rbf_local_tuning_folds.csv", index=False)
    best_round2 = tuning[tuning["postprocess"].eq("round2")].iloc[0]

    lattice, class_counts = target_lattice(train_df)
    lattice.to_csv(REPORTS_DIR / "v61_target_lattice_analysis.csv", index=False)
    class_counts.to_csv(REPORTS_DIR / "v61_y100_class_counts.csv", index=False)

    base_oof = tuning_oof[(tuning_oof["candidate"].eq("c1_g1_e0")) & (tuning_oof["postprocess"].eq("round2"))].copy()
    if base_oof.empty:
        base_oof = tuning_oof[tuning_oof["postprocess"].eq("round2")].copy().head(len(train_df))

    symbolic, terms = symbolic_approximation(train_df, base_oof)
    symbolic.to_csv(REPORTS_DIR / "v61_symbolic_formula_approximation.csv", index=False)
    terms.to_csv(REPORTS_DIR / "v61_sparse_formula_terms.csv", index=False)

    cf = counterfactual_probing(train_df)
    cf.to_csv(REPORTS_DIR / "v61_counterfactual_probing.csv", index=False)

    geom = sample_geometry(train_df, base_oof)
    geom.to_csv(REPORTS_DIR / "v61_sample_geometry_summary.csv", index=False)

    distill, rules, importances = rbf_distillation(train_df, base_oof)
    distill.to_csv(REPORTS_DIR / "v61_rbf_distillation_fidelity.csv", index=False)
    rules.to_csv(REPORTS_DIR / "v61_rbf_distillation_rules.csv", index=False)
    importances.to_csv(REPORTS_DIR / "v61_rbf_distillation_feature_importance.csv", index=False)

    clusters, profile = residual_topology(train_df, base_oof)
    clusters.to_csv(REPORTS_DIR / "v61_residual_topology_clusters.csv", index=False)
    profile.to_csv(REPORTS_DIR / "v61_large_residual_profile.csv", index=False)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = None
    if best_round2["mean_mae"] <= BASELINE_CV - 0.0005:
        path = maybe_save_submission(train_df, pd.read_csv(TEST_PATH), pd.read_csv(SAMPLE_SUBMISSION_PATH), best_round2)

    final_synthesis(best_round2, lattice, symbolic, geom, distill, clusters, path is not None)

    print("\n=== V6.1 tuning top ===")
    print(tuning.head(10).round(6).to_string(index=False))
    print("\n=== V6.1 lattice ===")
    print(lattice.head(12).to_string(index=False))
    print("\n=== V6.1 distillation ===")
    print(distill.round(6).to_string(index=False))
    if path:
        print(f"Saved submission: {path}")
    else:
        print("No v6.1 submission: local tuning did not clear the 0.0005 improvement threshold.")


if __name__ == "__main__":
    run_v61_experiments()
