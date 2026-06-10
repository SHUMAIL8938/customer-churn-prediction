"""
train.py
Trains Logistic Regression, Random Forest, and XGBoost models.
Saves the best model based on ROC-AUC.
"""

import os
import joblib
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from src.preprocess import preprocess


RANDOM_STATE = 42


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def tune_xgboost(X_train, y_train):
    """Quick RandomizedSearch to tune XGBoost."""
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best XGBoost params: {search.best_params_}")
    print(f"  Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def train():
    print("=" * 50)
    print("CUSTOMER CHURN PREDICTION — TRAINING")
    print("=" * 50)

    X, y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    os.makedirs("models", exist_ok=True)

    # --- Train baseline models ---
    models = get_models()
    cv = StratifiedKFold(n_splits=5)
    results = {}

    print("\n--- Cross-Validation (ROC-AUC) ---")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = scores.mean()
        print(f"  {name:<25} {scores.mean():.4f} ± {scores.std():.4f}")

    # --- Tune XGBoost ---
    print("\n--- Tuning XGBoost ---")
    best_xgb = tune_xgboost(X_train, y_train)
    best_xgb.fit(X_train, y_train)

    # Save all models
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.pkl")

    joblib.dump(best_xgb, "models/xgboost_tuned.pkl")

    # Save feature names for inference
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    print("\n✓ All models saved to models/")
    print("✓ Feature names saved to models/feature_names.pkl")

    # Return split for evaluation
    return X_train, X_test, y_train, y_test, best_xgb


if __name__ == "__main__":
    train()
