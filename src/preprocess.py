"""
preprocess.py
Handles data cleaning and feature engineering for the IBM Telco Kaggle dataset.
Columns match the 33-variable version with spaces in column names.

Download from Kaggle:
  https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
Save the CSV as: data/churn.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


# ── Column name map: Kaggle names → clean internal names ──────────────────────
RENAME = {
    "CustomerID":       "customerID",
    "Customer ID":      "customerID",
    "Gender":           "gender",
    "Senior Citizen":   "SeniorCitizen",
    "Partner":          "Partner",
    "Dependents":       "Dependents",
    "Tenure Months":    "tenure",
    "Phone Service":    "PhoneService",
    "Multiple Lines":   "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security":  "OnlineSecurity",
    "Online Backup":    "OnlineBackup",
    "Device Protection":"DeviceProtection",
    "Tech Support":     "TechSupport",
    "Streaming TV":     "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract":         "Contract",
    "Paperless Billing":"PaperlessBilling",
    "Payment Method":   "PaymentMethod",
    "Monthly Charge":   "MonthlyCharges",
    "Monthly Charges":  "MonthlyCharges",
    "Total Charges":    "TotalCharges",
    "Churn Label":      "Churn",
    "Churn Value":      "ChurnValue",
    "Churn Score":      "ChurnScore",
    "CLTV":             "CLTV",
    "Churn Reason":     "ChurnReason",
}

# Columns to drop — location info, leaky columns, IDs
DROP_COLS = [
    "customerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "ChurnValue",   # same as target, would leak
    "ChurnScore",   # IBM's own churn predictor — leakage
    "CLTV",         # derived from churn, leakage
    "ChurnReason",  # only known for churners, leakage
]

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PaperlessBilling",
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CAT_COLS = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_raw(path: str = "data/churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    # Rename to standard internal names
    df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns}, inplace=True)
    print(f"  Columns after rename: {list(df.columns)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop leaky / non-feature columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)
    print(f"  Dropped {len(drop)} columns: {drop}")

    # SeniorCitizen: "Yes"/"No" → 1/0 (Kaggle version uses strings)
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = (df["SeniorCitizen"].str.strip().str.lower() == "yes").astype(int)

    # TotalCharges may be string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    if missing_tc:
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["tenure"] * df["MonthlyCharges"]
        )
        print(f"  Imputed {missing_tc} missing TotalCharges values.")

    # Target: Churn label → 0/1
    if "Churn" in df.columns:
        churn_dtype = str(df["Churn"].dtype)
        if churn_dtype in ("object", "string") or "str" in churn_dtype:
            df["Churn"] = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
        else:
            df["Churn"] = df["Churn"].astype(int)

    dups = df.duplicated().sum()
    if dups:
        df.drop_duplicates(inplace=True)
        print(f"  Dropped {dups} duplicate rows.")

    print(f"  Churn rate: {df['Churn'].mean():.1%}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str).str.strip())

    # One-hot encode multi-category columns
    cats = [c for c in CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cats, drop_first=False)

    return df


def scale(df: pd.DataFrame, scaler_path: str = "models/scaler.pkl", fit: bool = True) -> pd.DataFrame:
    df = df.copy()
    num_cols = [c for c in NUM_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[num_cols] = scaler.transform(df[num_cols])

    return df


def preprocess(path: str = "data/churn.csv", fit_scaler: bool = True):
    df = load_raw(path)
    df = clean(df)
    df = encode(df)
    df = scale(df, fit=fit_scaler)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y


if __name__ == "__main__":
    X, y = preprocess()
    print("\nFeature columns:")
    for c in X.columns:
        print(f"  {c}")
