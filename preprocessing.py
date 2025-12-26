"""
preprocessing.py

Utility functions for loading data, building preprocessing pipelines and
splitting train/validation sets for the car insurance claim prediction project.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT  # train.csv and test.csv live here


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a CSV from the project data directory.

    Example: df = load_dataset("train.csv")
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file at {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Feature engineering / preprocessing
# ---------------------------------------------------------------------
def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Given a feature-only dataframe X (no target, no ID), return lists of
    numerical and categorical column names.
    """
    # Categorical = object dtype
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Numerical = int/float/bool
    numerical_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric features and one-hot encodes
    categorical ones.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def build_full_pipeline(
    preprocessor: ColumnTransformer,
    model,
) -> Pipeline:
    """
    Attach a model to the preprocessing pipeline.
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


# ---------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    target_col: str = "is_claim",
    id_col: str = "policy_id",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the *raw* dataframe into train/test sets.

    - Drops the target and id from X.
    - Stratifies by the target column.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")

    # Features: drop target and ID
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
