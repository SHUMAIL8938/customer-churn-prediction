"""
explain.py
Explainability using XGBoost built-in feature importance + permutation importance.
No SHAP required — zero compilation dependencies.

Three plots produced:
  1. feature_importance_gain.png   — which features the model splits on most
  2. feature_importance_permutation.png — how much accuracy drops when each feature is shuffled
  3. prediction_breakdown.png      — per-customer score breakdown (manual SHAP-style)
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
PLOTS_DIR = "plots"
TOP_N = 15


def plot_xgb_importance(model, feature_names, save_dir):
    """XGBoost built-in importance by 'gain' — how much each feature improves splits."""
    scores = model.get_booster().get_score(importance_type="gain")
    # Map internal feature names (f0, f1...) back to real names
    # XGBoost sometimes uses f0..fN so map via index
    try:
        pairs = [(feature_names[int(k[1:])], v) if k.startswith("f") else (k, v)
                 for k, v in scores.items()]
    except (ValueError, IndexError):
        pairs = list(scores.items())

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:TOP_N]
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#DC2626" if "Contract_Month" in n or "tenure" in n or "MonthlyCharges" in n
              else "#2563EB" for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Gain (higher = more important)", fontsize=11)
    ax.set_title("Feature Importance — XGBoost Gain", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    red_patch = mpatches.Patch(color="#DC2626", alpha=0.85, label="Top churn drivers")
    blue_patch = mpatches.Patch(color="#2563EB", alpha=0.85, label="Other features")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "feature_importance_gain.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: feature_importance_gain.png")
    return pairs


def plot_permutation_importance(model, X_test, y_test, save_dir):
    """
    Permutation importance: shuffle each feature and measure how much
    ROC-AUC drops. More reliable than built-in importance for real impact.
    """
    print("  Computing permutation importance (takes ~20s) ...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    idx = np.argsort(result.importances_mean)[::-1][:TOP_N]
    names = [X_test.columns[i] for i in idx]
    means = result.importances_mean[idx]
    stds = result.importances_std[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#DC2626" if v > 0.005 else "#94A3B8" for v in means]
    ax.barh(range(len(names)), means, xerr=stds, color=colors,
            alpha=0.85, edgecolor="white", capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ROC-AUC drop when shuffled", fontsize=11)
    ax.set_title("Permutation Importance (ROC-AUC)", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "feature_importance_permutation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: feature_importance_permutation.png")
    return list(zip(names, means))


def plot_prediction_breakdown(model, X_test, y_test, feature_names, save_dir):
    """
    Manual SHAP-style breakdown for one high-risk customer.
    Shows feature value + model's marginal contribution per feature
    using the difference from base rate.
    """
    # Find a high-confidence churner
    probs = model.predict_proba(X_test)[:, 1]
    churner_idx = np.where(y_test.values == 1)[0]
    best = churner_idx[np.argmax(probs[churner_idx])]
    sample = X_test.iloc[best]
    prob = probs[best]

    # Get feature importances as proxy weights
    scores = model.get_booster().get_score(importance_type="gain")
    try:
        fi = {feature_names[int(k[1:])]: v if k.startswith("f") else v
              for k, v in scores.items()}
    except (ValueError, IndexError):
        fi = dict(scores)

    base_rate = y_test.mean()

    # Build contributions: importance × (feature_value − feature_mean)
    # Normalise so they sum to (prob − base_rate)
    X_arr = X_test.values
    feat_means = X_arr.mean(axis=0)
    contributions = {}
    for i, fname in enumerate(feature_names):
        w = fi.get(fname, 0)
        contributions[fname] = w * (sample.iloc[i] - feat_means[i])

    total = sum(contributions.values()) or 1
    scale = (prob - base_rate) / total
    contributions = {k: v * scale for k, v in contributions.items()}

    top = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [t[0] for t in top]
    vals = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Contribution to churn probability", fontsize=11)
    ax.set_title(
        f"Why this customer was flagged  (predicted churn prob: {prob:.0%})",
        fontsize=12, fontweight="bold"
    )

    red_patch = mpatches.Patch(color="#DC2626", alpha=0.85, label="Increases churn risk")
    blue_patch = mpatches.Patch(color="#2563EB", alpha=0.85, label="Decreases churn risk")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "prediction_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: prediction_breakdown.png")


def explain():
    print("=" * 50)
    print("CUSTOMER CHURN — EXPLAINABILITY (no SHAP)")
    print("=" * 50)

    X, y = preprocess(fit_scaler=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_path = "models/xgboost_tuned.pkl"
    if not os.path.exists(model_path):
        print("  Model not found. Run train.py first.")
        return

    model = joblib.load(model_path)
    feature_names = list(X.columns)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n--- XGBoost built-in importance (gain) ---")
    top_gain = plot_xgb_importance(model, feature_names, PLOTS_DIR)

    print("\n--- Permutation importance ---")
    top_perm = plot_permutation_importance(model, X_test, y_test, PLOTS_DIR)

    print("\n--- Single-customer prediction breakdown ---")
    plot_prediction_breakdown(model, X_test, y_test, feature_names, PLOTS_DIR)

    print("\n  Top 5 features (gain):")
    for name, score in top_gain[:5]:
        print(f"    {name:<40} {score:>10.2f}")

    print("\n  Top 5 features (permutation ROC-AUC drop):")
    for name, score in top_perm[:5]:
        print(f"    {name:<40} {score:>10.4f}")

    print("\n✓ Explainability complete. Plots saved to plots/")


if __name__ == "__main__":
    explain()
