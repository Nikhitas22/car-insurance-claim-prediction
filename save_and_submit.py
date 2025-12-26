"""
save_and_submit.py

Thin wrapper around predictions.py if you want a Kaggle-style 'submission.csv'.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictions import main as run_predictions

PROJECT_ROOT = Path(__file__).resolve().parent
SUB_PATH = PROJECT_ROOT / "submission.csv"
PRED_PATH = PROJECT_ROOT / "predictions.csv"


def main():
    # Generate predictions.csv
    run_predictions()

    pred_df = pd.read_csv(PRED_PATH)
    # For Kaggle: usually 'policy_id' + 'is_claim'
    if "is_claim_probability" in pred_df.columns:
        pred_df = pred_df.rename(columns={"is_claim_probability": "is_claim"})

    pred_df.to_csv(SUB_PATH, index=False)
    print(f"✅ Saved submission file to {SUB_PATH}")


if __name__ == "__main__":
    main()
