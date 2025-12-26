import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_pipeline.joblib"
OUTPUT_PATH = PROJECT_ROOT / "feature_importance.csv"

def main():
    pipeline = joblib.load(MODEL_PATH)

    # Get model
    model = pipeline.named_steps["model"]

    # Get feature names safely
    preprocessor = pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
    feature_names = preprocessor.get_feature_names_out()

    # Extract importance
    importances = model.feature_importances_

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    df_importance.to_csv(OUTPUT_PATH, index=False)
    print("✅ Feature importance saved to feature_importance.csv")

if __name__ == "__main__":
    main()
