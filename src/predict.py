"""
predict.py
Single-customer inference using the trained XGBoost model.
Matches the real Kaggle IBM Telco dataset feature space.
"""

from dataclasses import field
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = "models"


def load_artifacts():
    model        = joblib.load(os.path.join(MODELS_DIR, "xgboost_tuned.pkl"))
    scaler       = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


def build_input_df(customer: dict, feature_names: list) -> pd.DataFrame:
    """
    Convert a raw customer dict into an encoded DataFrame that matches
    the training feature space exactly.
    """
    def yes_no(field):
        return 1 if customer.get(field) in ("Yes", 1, True) else 0

    row = {
        "SeniorCitizen":    int(customer.get("senior_citizen")or 0),
        "tenure":           float(customer["tenure"]),
        "MonthlyCharges":   float(customer["monthly_charges"]),
        "TotalCharges":     float(customer["total_charges"] 
                        if customer.get("total_charges") is not None
                        else customer["tenure"] * customer["monthly_charges"]),
        "gender":           1 if customer.get("gender", "Male") == "Male" else 0,
        "Partner":          yes_no("partner"),
        "Dependents":       yes_no("dependents"),
        "PaperlessBilling": yes_no("paperless_billing"),
        "PhoneService":     yes_no("phone_service"),
        "OnlineSecurity":   yes_no("online_security"),
        "OnlineBackup":     yes_no("online_backup"),
        "DeviceProtection": yes_no("device_protection"),
        "TechSupport":      yes_no("tech_support"),
        "StreamingTV":      yes_no("streaming_tv"),
        "StreamingMovies":  yes_no("streaming_movies"),
    }

    # One-hot: MultipleLines
    for val in ["No", "No phone service", "Yes"]:
        row[f"MultipleLines_{val}"] = 1 if customer.get("multiple_lines") == val else 0

    # One-hot: InternetService
    for val in ["Cable", "DSL", "Fiber Optic", "No"]:
        row[f"InternetService_{val}"] = 1 if customer.get("internet_service") == val else 0

    # One-hot: Contract
    for val in ["Month-to-Month", "One Year", "Two Year"]:
        row[f"Contract_{val}"] = 1 if customer.get("contract") == val else 0

    # One-hot: PaymentMethod
    for val in ["Bank Withdrawal", "Credit Card", "Mailed Check"]:
        row[f"PaymentMethod_{val}"] = 1 if customer.get("payment_method") == val else 0

    # Align with exact training columns (fill any missing with 0)
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df


def predict(customer: dict) -> dict:
    model, scaler, feature_names = load_artifacts()
    df = build_input_df(customer, feature_names)

    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
                if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    prob  = model.predict_proba(df)[0][1]
    label = int(prob >= 0.5)

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction":  label,
        "risk_level":        "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
    }


if __name__ == "__main__":
    sample = {
        "tenure": 2,
        "monthly_charges": 95.0,
        "senior_citizen": 0,
        "gender": "Male",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber Optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-Month",
        "paperless_billing": "Yes",
        "payment_method": "Mailed Check",
    }
    print(predict(sample))
