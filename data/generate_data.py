"""
generate_data.py
Generates a synthetic IBM Telco-style churn dataset for reproducibility.
Run this if you don't have the original CSV:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043


def generate_churn_dataset(n=N):
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
    )
    tenure = np.where(
        contract == "Month-to-month",
        np.random.randint(1, 30, n),
        np.where(contract == "One year", np.random.randint(12, 48, n), np.random.randint(24, 72, n)),
    )

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    monthly_charges = np.where(
        internet_service == "Fiber optic",
        np.random.uniform(70, 110, n),
        np.where(internet_service == "DSL", np.random.uniform(40, 75, n), np.random.uniform(18, 30, n)),
    ).round(2)

    total_charges = (tenure * monthly_charges + np.random.normal(0, 50, n)).clip(0).round(2)
    # Inject a few blanks (like the real dataset)
    blank_idx = np.random.choice(n, 11, replace=False)
    total_charges_str = total_charges.astype(str)
    total_charges_str[blank_idx] = " "

    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Churn probability driven by real correlations
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet_service == "Fiber optic")
        + 0.15 * (tenure < 6)
        - 0.10 * (tenure > 36)
        + 0.08 * (monthly_charges > 80)
        + 0.06 * (senior == 1)
        - 0.05 * (partner == "Yes")
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)

    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    online_security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.50, 0.21])
    tech_support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.50, 0.21])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "InternetService": internet_service,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn_label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data/churn.csv", index=False)
    print(f"Dataset generated: {len(df)} rows, {df.columns.tolist()}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
