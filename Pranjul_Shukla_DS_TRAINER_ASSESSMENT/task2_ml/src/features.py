"""Leakage-aware sklearn preprocessing for the cleaned FinTrust records."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import TARGET_COLUMN, get_predictor_columns


def to_dense_array(matrix: Any) -> np.ndarray:
    """Convert sparse preprocessor output for estimators requiring dense input."""
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing steps; fitting is deferred to the training fold."""
    if features.empty:
        raise ValueError("Cannot build a preprocessor from an empty feature table.")
    columns = get_predictor_columns(features)
    if not columns:
        raise ValueError("No predictive feature columns remain after leakage exclusions.")

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Combine the fold-fitted preprocessing transformer and a chosen estimator."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("estimator", estimator),
        ]
    )


def feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """Return encoded feature names after the preprocessor has been fitted."""
    return preprocessor.get_feature_names_out().tolist()


def predictor_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Select eligible predictors while keeping identifiers/labels outside X."""
    columns = get_predictor_columns(data)
    if TARGET_COLUMN in columns:
        raise ValueError("Target column cannot be included in the predictor frame.")
    return data.loc[:, columns].copy()
