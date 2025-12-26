"""
eda.py

Modular EDA functions for Car Insurance Claim Prediction.
Follows PEP8 conventions, clean structure, and documentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def get_feature_types(df: pd.DataFrame):
    """Return numerical and categorical columns."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_cols, categorical_cols


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print missing value summary."""
    missing = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100

    summary = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percent
    })

    print(summary[summary["Missing Count"] > 0])
    return summary


def plot_missing_values(df: pd.DataFrame):
    """Plot missing value percentage chart."""
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent[missing_percent > 0]

    if len(missing_percent) == 0:
        print("No missing values in dataset.")
        return

    plt.figure(figsize=(12, 5))
    sns.barplot(x=missing_percent.values, y=missing_percent.index)
    plt.title("Missing Value Percentage")
    plt.xlabel("% Missing")
    plt.ylabel("Columns")
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list):
    """Plot histograms for numeric columns."""
    df[numerical_cols].hist(figsize=(15, 15), bins=30)
    plt.suptitle("Numerical Feature Distributions")
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame, categorical_cols: list):
    """Plot bar charts for categorical variables."""
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numerical_cols: list):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_target_distribution(df: pd.DataFrame):
    """Plot distribution of target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_claim", data=df)
    plt.title("Target Variable Distribution")
    plt.show()
