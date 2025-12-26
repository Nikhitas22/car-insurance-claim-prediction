"""
predictions.py

Use the tuned pipeline (final_pipeline.joblib) to generate predictions for
test.csv. Outputs predictions.csv with:
- policy_id
- is_claim_probability
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocessing import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUT_PATH = PROJECT_ROOT / "predictions.csv"


def main():
    model_path = MODELS_DIR / "final_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run hyperparameter_tuning.py first."
        )

    pipe = joblib.load(model_path)
    test_df = load_dataset("test.csv")

    if "policy_id" not in test_df.columns:
        raise ValueError("Expected 'policy_id' column in test.csv")

    preds_proba = pipe.predict_proba(test_df.drop(columns=["policy_id"]))[:, 1]

    out_df = pd.DataFrame(
        {
            "policy_id": test_df["policy_id"],
            "is_claim_probability": preds_proba,
        }
    )

    out_df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved predictions to {OUT_PATH}")


if __name__ == "__main__":
    main()
