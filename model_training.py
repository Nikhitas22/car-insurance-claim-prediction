"""
model_training.py

Train baseline and advanced models on the car insurance dataset and compare
their performance.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

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


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    # Some models may not support predict_proba (e.g., LinearSVC)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        roc_auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n========== {name} ==========")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def main():
    df = load_dataset("train.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Determine feature types from training data (no target, no ID)
    numerical_cols, categorical_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Linear SVC": LinearSVC(random_state=42),
    }

    results = []

    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_full_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)

        # Use ROC-AUC when available, else F1
        score = metrics["roc_auc"]
        if np.isnan(score):
            score = metrics["f1"]

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    print("\n======= FINAL MODEL COMPARISON =======")
    for r in results:
        print(
            f"{r['model']}: "
            f"Accuracy={r['accuracy']:.4f}, "
            f"F1={r['f1']:.4f}, "
            f"ROC-AUC={r['roc_auc']:.4f}"
        )

    if best_pipeline is not None:
        out_path = MODELS_DIR / "baseline_pipeline.joblib"
        joblib.dump(best_pipeline, out_path)
        print(f"\n✅ Saved best baseline pipeline ({best_name}) to {out_path}")


if __name__ == "__main__":
    main()
