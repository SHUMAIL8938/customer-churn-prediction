# Customer Churn Prediction with Explainable AI

End-to-end ML project predicting telecom customer churn with explainability and a REST API.

---

## Dataset Setup (Real Kaggle Data)

1. Go to: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
2. Download `Telco_customer_churn.csv`
3. Rename it to `churn.csv` and place it in the `data/` folder

> No Kaggle account? Run `python data/generate_data.py` to use synthetic data instead.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (data → train → evaluate → explain)
python run_pipeline.py

# 3. Start the API
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── churn.csv                     ← place Kaggle CSV here
│   └── generate_data.py              ← synthetic fallback
├── notebooks/
│   └── eda.ipynb                     ← EDA with 6 plots
├── src/
│   ├── preprocess.py                 ← cleaning + encoding + scaling
│   ├── train.py                      ← 3 models + hyperparameter tuning
│   ├── evaluate.py                   ← 5 metrics + plots
│   ├── explain.py                    ← feature importance (no SHAP needed)
│   └── predict.py                    ← single-customer inference
├── api/
│   └── main.py                       ← FastAPI REST endpoint
├── models/                           ← saved after training
├── plots/                            ← saved after training
├── run_pipeline.py                   ← runs everything end-to-end
└── requirements.txt
```

---

## What the Real Dataset Adds

The Kaggle dataset has 33 columns vs the standard 21. Key extras used:

| Column | What it adds |
|---|---|
| Phone Service, Multiple Lines | Phone plan features |
| Online Backup, Device Protection | Add-on services |
| Streaming TV, Streaming Movies | Usage behavior |
| Churn Reason | Only used in EDA (dropped before training — would leak) |
| Churn Score, CLTV | Dropped — IBM's own derived predictions, would leak |

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "monthly_charges": 95.0,
    "internet_service": "Fiber Optic",
    "contract": "Month-to-Month",
    "payment_method": "Mailed Check",
    "online_security": "No",
    "tech_support": "No"
  }'
```

Response:
```json
{
  "churn_probability": 0.8412,
  "churn_prediction": 1,
  "risk_level": "High"
}
```

---

## Model Performance (Typical on Real Data)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.80 | ~0.58 | ~0.84 |
| Random Forest | ~0.81 | ~0.61 | ~0.85 |
| **XGBoost (Tuned)** | **~0.82** | **~0.63** | **~0.87** |

---

## Explainability (no SHAP required)

Three plots in `plots/`:
- **feature_importance_gain.png** — XGBoost split gain per feature
- **feature_importance_permutation.png** — ROC-AUC drop when feature is shuffled
- **prediction_breakdown.png** — why a specific customer was flagged

Top churn drivers on the real dataset: Contract type → Tenure → Monthly Charges → Internet Service → Tech Support

---


