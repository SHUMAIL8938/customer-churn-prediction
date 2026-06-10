"""
evaluate.py
Evaluates all trained models and generates comparison plots.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from src.preprocess import preprocess
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
MODELS_DIR = "models"
PLOTS_DIR = "plots"


def load_models():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost (Tuned)": "xgboost_tuned.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }, y_pred, y_prob


def plot_roc_curves(models, X_test, y_test, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=150)
    plt.close()
    print("  Saved: roc_curves.png")


def plot_confusion_matrix(model, X_test, y_test, model_name, save_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(os.path.join(save_dir, f"cm_{safe_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved: cm_{safe_name}.png")


def plot_metrics_comparison(results, save_dir):
    import pandas as pd

    df = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df.columns))
    width = 0.25
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    for i, (model_name, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, row.values, width, label=model_name, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: metrics_comparison.png")


def evaluate():
    print("=" * 50)
    print("CUSTOMER CHURN PREDICTION — EVALUATION")
    print("=" * 50)

    X, y = preprocess(fit_scaler=False)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = load_models()
    if not models:
        print("No models found. Run train.py first.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    results = {}
    print("\n--- Evaluation Metrics ---")
    print(f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
    print("-" * 73)

    for name, model in models.items():
        metrics, y_pred, y_prob = compute_metrics(model, X_test, y_test)
        results[name] = metrics
        print(
            f"  {name:<23} {metrics['Accuracy']:>9.4f} {metrics['Precision']:>10.4f} "
            f"{metrics['Recall']:>8.4f} {metrics['F1 Score']:>8.4f} {metrics['ROC-AUC']:>9.4f}"
        )

    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])
    print(f"\n  Best model (ROC-AUC): {best_model_name}")

    print("\n--- Generating Plots ---")
    plot_roc_curves(models, X_test, y_test, PLOTS_DIR)
    plot_metrics_comparison(results, PLOTS_DIR)

    best_model = models[best_model_name]
    plot_confusion_matrix(best_model, X_test, y_test, best_model_name, PLOTS_DIR)

    print(f"\n  Detailed report for {best_model_name}:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=["Stay", "Churn"]))

    return results


if __name__ == "__main__":
    evaluate()
