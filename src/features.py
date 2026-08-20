import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BASE_CATEGORICAL_COLS,
    CATEGORICAL_MISSING_COLS,
    INTERACTION_SPECS,
    RAW_EXCLUDED_FEATURES,
    SCORECARD_BIN_SPECS,
    V2_CATEGORICAL_COLS,
    V3_CATEGORICAL_COLS,
)


class StressFeatureEngineer(BaseEstimator, TransformerMixin):
    """Train-only fitted feature engineering for the stress baseline."""

    def __init__(self, missing_token="__MISSING__", add_mean_working_cat_v3=False):
        self.missing_token = missing_token
        self.add_mean_working_cat_v3 = add_mean_working_cat_v3

    def fit(self, X, y=None):
        X_df = self._to_frame(X)
        self.mean_working_median_ = X_df["mean_working"].median()
        return self

    def transform(self, X):
        X_df = self._to_frame(X).copy()

        for col in CATEGORICAL_MISSING_COLS:
            X_df[col] = X_df[col].astype("object").fillna(self.missing_token)

        mean_working = X_df["mean_working"]
        X_df["mean_working_missing"] = mean_working.isna().astype("int8")
        X_df["mean_working_imputed"] = mean_working.fillna(self.mean_working_median_)
        X_df["mean_working_cat"] = self._mean_working_category(mean_working)
        if self.add_mean_working_cat_v3:
            X_df["mean_working_cat_v3"] = self._mean_working_category_v3(mean_working)

        height_m = X_df["height"] / 100.0
        X_df["bmi"] = X_df["weight"] / np.square(height_m)
        X_df["pulse_pressure"] = (
            X_df["systolic_blood_pressure"] - X_df["diastolic_blood_pressure"]
        )
        X_df["map"] = X_df["diastolic_blood_pressure"] + X_df["pulse_pressure"] / 3.0

        return X_df.drop(columns=[c for c in RAW_EXCLUDED_FEATURES if c in X_df.columns])

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    @staticmethod
    def _mean_working_category(values):
        return np.select(
            [
                values.isna(),
                values <= 6,
                values.between(7, 8, inclusive="both"),
                values.between(9, 10, inclusive="both"),
                values >= 11,
            ],
            [
                "missing",
                "low_<=6",
                "standard_7_8",
                "extended_9_10",
                "high_11plus",
            ],
            default="extended_9_10",
        )

    @staticmethod
    def _mean_working_category_v3(values):
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
            [
                "missing",
                "low_<=6",
                "work_7",
                "work_8",
                "work_9",
                "work_10",
                "work_11",
                "work_12plus",
            ],
            default="work_12plus",
        )


def make_preprocessor(scale_numeric=False):
    return make_ohe_preprocessor(BASE_CATEGORICAL_COLS, scale_numeric=scale_numeric)


def make_ohe_preprocessor(categorical_cols, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    from sklearn.compose import ColumnTransformer

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class QuantileScorecardBinner(BaseEstimator, TransformerMixin):
    """Fold-fitted qcut-style binning that never looks at validation/test distribution."""

    def __init__(self, bin_specs=None, n_bins=5, missing_token="__MISSING__"):
        self.bin_specs = bin_specs
        self.n_bins = n_bins
        self.missing_token = missing_token

    def fit(self, X, y=None):
        X_df = self._to_frame(X)
        specs = self.bin_specs or SCORECARD_BIN_SPECS
        self.edges_ = {}
        for source_col in specs:
            values = pd.to_numeric(X_df[source_col], errors="coerce").dropna()
            if values.nunique() <= 1:
                edges = np.array([-np.inf, np.inf], dtype=float)
            else:
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                edges = values.quantile(quantiles).to_numpy(dtype=float)
                edges = np.unique(edges)
                edges[0] = -np.inf
                edges[-1] = np.inf
                if len(edges) < 2:
                    edges = np.array([-np.inf, np.inf], dtype=float)
            self.edges_[source_col] = edges
        return self

    def transform(self, X):
        X_df = self._to_frame(X).copy()
        specs = self.bin_specs or SCORECARD_BIN_SPECS
        for source_col, bin_col in specs.items():
            values = pd.to_numeric(X_df[source_col], errors="coerce")
            labels = [f"bin_{i:02d}" for i in range(len(self.edges_[source_col]) - 1)]
            X_df[bin_col] = pd.cut(
                values,
                bins=self.edges_[source_col],
                labels=labels,
                include_lowest=True,
                duplicates="drop",
            ).astype("object")
            X_df[bin_col] = X_df[bin_col].fillna(self.missing_token)
        return X_df

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, interaction_specs=None, missing_token="__MISSING__"):
        self.interaction_specs = interaction_specs
        self.missing_token = missing_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = self._to_frame(X).copy()
        specs = self.interaction_specs or INTERACTION_SPECS
        for left, right, out_col in specs:
            left_values = X_df[left].astype("object").fillna(self.missing_token).astype(str)
            right_values = X_df[right].astype("object").fillna(self.missing_token).astype(str)
            X_df[out_col] = left_values + "__x__" + right_values
        return X_df

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class CategoricalStringCaster(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols=None, missing_token="__MISSING__"):
        self.categorical_cols = categorical_cols
        self.missing_token = missing_token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = self._to_frame(X).copy()
        categorical_cols = self.categorical_cols or V3_CATEGORICAL_COLS
        for col in categorical_cols:
            X_df[col] = X_df[col].astype("object").fillna(self.missing_token).astype(str)
        return X_df

    @staticmethod
    def _to_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
