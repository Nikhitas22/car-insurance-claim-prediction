"""
hyperparameter_tuning.py

GridSearchCV for RandomForest using ROC-AUC.
Best pipeline is saved to 'models/final_pipeline.joblib'.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessing import (
    build_full_pipeline,
    build_preprocessor,
    get_feature_types,
    load_dataset,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    # Always tune on the original training data
    df = load_dataset("train.csv")

    # Split FIRST so features don't contain target/id
    X_train, X_test, y_train, y_test = split_data(df)

    # Get feature types from feature-only dataframe
    numerical_cols, categorical_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    base_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline_rf = build_full_pipeline(preprocessor, base_model)

    param_grid = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="roc_auc",
        verbose=2,
    )

    print("Starting GridSearchCV (RandomForest, scoring=ROC-AUC)...")
    grid.fit(X_train, y_train)

    print("\nBest Params:", grid.best_params_)
    print("Best ROC-AUC (CV):", grid.best_score_)

    best_pipeline = grid.best_estimator_

    out_path = MODELS_DIR / "final_pipeline.joblib"
    joblib.dump(best_pipeline, out_path)
    print(f"\n✅ Saved tuned RandomForest pipeline to: {out_path}")


if __name__ == "__main__":
    main()
