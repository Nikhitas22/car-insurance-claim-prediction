"""
evaluation.py

Evaluate the tuned model, export:
- processed_data.csv  (features + target used for modelling)
- feature_importance.csv (for RandomForest)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from preprocessing import load_dataset, split_data

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUT_PROCESSED = PROJECT_ROOT / "processed_data.csv"
OUT_FEATURES = PROJECT_ROOT / "feature_importance.csv"


def main():
    df = load_dataset("train.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    model_path = MODELS_DIR / "final_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run hyperparameter_tuning.py first."
        )

    pipe = joblib.load(model_path)

    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        roc_auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== FINAL MODEL EVALUATION =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # Export processed data (for Power BI)
    # ------------------------------------------------------------------
    processed = X_test.copy()
    processed["is_claim"] = y_test.values
    if y_proba is not None:
        processed["predicted_proba"] = y_proba
    processed["predicted_label"] = y_pred
    processed.to_csv(OUT_PROCESSED, index=False)
    print(f"\n✅ Saved processed data to {OUT_PROCESSED}")

    # ------------------------------------------------------------------
    # Feature importance (works for tree models such as RandomForest)
    # ------------------------------------------------------------------
    model = pipe.named_steps.get("model")
    if hasattr(model, "feature_importances_"):
        # ColumnTransformer + OneHot => we can't easily get expanded names here
        # but we can at least export raw importances and use them for analysis.
        importances = model.feature_importances_
        fi = pd.DataFrame(
            {
                "feature_index": np.arange(len(importances)),
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        fi.to_csv(OUT_FEATURES, index=False)
        print(f"✅ Saved feature importances to {OUT_FEATURES}")
    else:
        print("⚠️ Model does not expose feature_importances_. Skipping export.")


if __name__ == "__main__":
    main()
