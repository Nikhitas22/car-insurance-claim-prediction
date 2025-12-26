"""
app_streamlit.py

Medium-complexity Streamlit app for Car Insurance Claim Prediction.

Assumptions:
- You have run model_training.py / hyperparameter_tuning.py
  and saved the best pipeline to: models/final_pipeline.joblib
- preprocessing.py defines load_dataset("train.csv")
- train.csv contains the columns:
  policy_id, policy_tenure, age_of_car, age_of_policyholder,
  area_cluster, population_density, make, segment, model,
  fuel_type, transmission_type, ncap_rating,
  and binary features like is_brake_assist, is_power_steering, etc.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import load_dataset


# -------------------------------------------------------------------
# Paths & cached loaders
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_pipeline.joblib"


@st.cache_resource
def load_pipeline(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}.\n"
            "Run model_training.py or hyperparameter_tuning.py first."
        )
    return joblib.load(path)


@st.cache_data
def load_training_X() -> pd.DataFrame:
    """Load train.csv and return feature matrix X (no policy_id, no target)."""
    df = load_dataset("train.csv")
    # your split_data() uses: X = df.drop(columns=[target_col, "policy_id"])
    cols_to_drop = [c for c in ["policy_id", "is_claim"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    return X


# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------

def safe_set(df: pd.DataFrame, col: str, value):
    """Set df.loc[0, col] only if column exists."""
    if col in df.columns:
        df.loc[0, col] = value


def yes_no_to_int(label: str) -> int:
    """Map 'Yes'/'No' (or True/False) to 1/0 integers."""
    if isinstance(label, bool):
        return int(label)
    return 1 if str(label).lower() in {"yes", "y", "1", "true"} else 0


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Car Insurance Claim Prediction",
    page_icon="🚗",
    layout="centered",
)


def main():
    st.title("🚗 Car Insurance Claim Prediction")
    st.write(
        "This app uses your trained machine-learning model to estimate the "
        "probability that a policy will result in a **claim at renewal**. "
        "Enter the key customer & vehicle details below and click **Predict**."
    )

    # Load model and training data
    pipeline = load_pipeline()
    train_X = load_training_X()
    template = train_X.iloc[[0]].copy()  # 1-row template with correct columns
    feature_cols = list(train_X.columns)

    # Convenience for selectbox options
    def cat_options(col: str):
        if col in train_X.columns:
            return sorted(train_X[col].dropna().unique().tolist())
        return []

    # ------------------------------------------------------------------
    # Layout – Customer & Policy block
    # ------------------------------------------------------------------
    st.subheader("1️⃣ Customer & Policy Information")
    col1, col2 = st.columns(2)

    with col1:
        age_of_policyholder = st.number_input(
            "Age of Policyholder (years)",
            value=int(train_X["age_of_policyholder"].median())
            if "age_of_policyholder" in train_X.columns
            else 30,
            step=1,
        )

        age_of_car = st.number_input(
            "Age of Car (years)",
            value=int(train_X["age_of_car"].median())
            if "age_of_car" in train_X.columns
            else 3,
            step=1,
        )

        policy_tenure = st.number_input(
            "Policy Tenure (months)",
            value=int(train_X["policy_tenure"].median())
            if "policy_tenure" in train_X.columns
            else 12,
            step=1,
        )

    with col2:
        area_cluster = st.selectbox(
            "Area Cluster",
            options=cat_options("area_cluster") or ["C1", "C2", "C3"],
        )

        population_density = st.number_input(
            "Population Density",
            value=float(train_X["population_density"].median())
            if "population_density" in train_X.columns
            else 3000.0,
            step=100.0,
        )

    # ------------------------------------------------------------------
    # Layout – Vehicle block
    # ------------------------------------------------------------------
    st.subheader("2️⃣ Vehicle Information")
    col3, col4 = st.columns(2)

    with col3:
        make = st.selectbox(
            "Vehicle Make",
            options=cat_options("make") or ["A", "B", "C"],
        )

        segment = st.selectbox(
            "Vehicle Segment",
            options=cat_options("segment") or ["A", "B", "C"],
        )

        model_name = st.selectbox(
            "Vehicle Model",
            options=cat_options("model") or ["M1", "M2", "M3"],
        )

    with col4:
        fuel_type = st.selectbox(
            "Fuel Type",
            options=cat_options("fuel_type") or ["Petrol", "Diesel", "CNG"],
        )

        transmission_type = st.selectbox(
            "Transmission Type",
            options=cat_options("transmission_type") or ["Manual", "Automatic"],
        )

        ncap_rating = st.number_input(
            "NCAP Rating (0–5)",
            value=int(train_X["ncap_rating"].median())
            if "ncap_rating" in train_X.columns
            else 3,
            step=1,
        )

    # ------------------------------------------------------------------
    # Layout – Safety features (binary yes/no)
    # ------------------------------------------------------------------
    st.subheader("3️⃣ Safety & Convenience Features (optional)")
    col5, col6 = st.columns(2)

    with col5:
        brake_assist = st.selectbox("Brake Assist", ["No", "Yes"])
        power_steering = st.selectbox("Power Steering", ["No", "Yes"])
        power_door_locks = st.selectbox("Power Door Locks", ["No", "Yes"])
        central_locking = st.selectbox("Central Locking", ["No", "Yes"])

    with col6:
        driver_seat_height_adj = st.selectbox(
            "Driver Seat Height Adjustable", ["No", "Yes"]
        )
        day_night_rvm = st.selectbox(
            "Day/Night Rear View Mirror", ["No", "Yes"]
        )
        ecw = st.selectbox("Emergency Cornering Warning (ECW)", ["No", "Yes"])
        speed_alert = st.selectbox("Speed Alert", ["No", "Yes"])

    st.markdown("---")

    # ------------------------------------------------------------------
    # Build feature row when user clicks Predict
    # ------------------------------------------------------------------
    if st.button("🔍 Predict Claim Probability"):
        X_new = template.copy()

        # Set numeric / categorical features
        safe_set(X_new, "age_of_policyholder", age_of_policyholder)
        safe_set(X_new, "age_of_car", age_of_car)
        safe_set(X_new, "policy_tenure", policy_tenure)
        safe_set(X_new, "area_cluster", area_cluster)
        safe_set(X_new, "population_density", population_density)
        safe_set(X_new, "make", make)
        safe_set(X_new, "segment", segment)
        safe_set(X_new, "model", model_name)
        safe_set(X_new, "fuel_type", fuel_type)
        safe_set(X_new, "transmission_type", transmission_type)
        safe_set(X_new, "ncap_rating", ncap_rating)

        # Map yes/no -> 0/1 for binary safety features
        safe_set(X_new, "is_brake_assist", yes_no_to_int(brake_assist))
        safe_set(X_new, "is_power_steering", yes_no_to_int(power_steering))
        safe_set(X_new, "is_power_door_locks", yes_no_to_int(power_door_locks))
        safe_set(X_new, "is_central_locking", yes_no_to_int(central_locking))
        safe_set(
            X_new,
            "is_driver_seat_height_adjustable",
            yes_no_to_int(driver_seat_height_adj),
        )
        safe_set(
            X_new,
            "is_day_night_rear_view_mirror",
            yes_no_to_int(day_night_rvm),
        )
        safe_set(X_new, "is_ecw", yes_no_to_int(ecw))
        safe_set(X_new, "is_speed_alert", yes_no_to_int(speed_alert))

        try:
            proba = pipeline.predict_proba(X_new)[:, 1][0]
            pred_label = int(pipeline.predict(X_new)[0])

            st.success("Prediction completed.")
            st.metric(
                "Predicted Claim Probability",
                f"{proba:.2%}",
                help="Probability that this policy results in a claim at renewal.",
            )
            st.write(
                f"**Predicted class (is_claim)**: `{pred_label}` "
                "(1 = claim, 0 = no claim)"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

            with st.expander("Debug information (for development)"):
                st.write("X_new columns:", list(X_new.columns))
                if hasattr(pipeline, "feature_names_in_"):
                    st.write("Model expects:", list(pipeline.feature_names_in_))
                st.write("First row sent to model:")
                st.dataframe(X_new)


if __name__ == "__main__":
    main()
