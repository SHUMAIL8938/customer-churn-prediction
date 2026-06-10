"""
main.py — FastAPI churn prediction API
Matches real IBM Telco Kaggle dataset fields.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict

app = FastAPI(
    title="Customer Churn Prediction API",
    description="IBM Telco dataset — predicts churn probability with risk level.",
    version="2.0.0",
)


class CustomerInput(BaseModel):
    tenure:             float = Field(..., ge=0, le=72,  description="Months with the company")
    monthly_charges:    float = Field(..., ge=0,          description="Monthly bill USD")
    total_charges:      Optional[float] = Field(None,     description="Total billed (auto if omitted)")
    senior_citizen:     Optional[int]   = Field(0,  ge=0, le=1)
    gender:             Optional[str]   = Field("Male",   description="Male | Female")
    partner:            Optional[str]   = Field("No",     description="Yes | No")
    dependents:         Optional[str]   = Field("No",     description="Yes | No")
    phone_service:      Optional[str]   = Field("Yes",    description="Yes | No")
    multiple_lines:     Optional[str]   = Field("No",     description="No | Yes | No phone service")
    internet_service:   Optional[str]   = Field("Fiber Optic", description="DSL | Fiber Optic | Cable | No")
    online_security:    Optional[str]   = Field("No",     description="Yes | No")
    online_backup:      Optional[str]   = Field("No",     description="Yes | No")
    device_protection:  Optional[str]   = Field("No",     description="Yes | No")
    tech_support:       Optional[str]   = Field("No",     description="Yes | No")
    streaming_tv:       Optional[str]   = Field("No",     description="Yes | No")
    streaming_movies:   Optional[str]   = Field("No",     description="Yes | No")
    contract:           Optional[str]   = Field("Month-to-Month", description="Month-to-Month | One Year | Two Year")
    paperless_billing:  Optional[str]   = Field("Yes",    description="Yes | No")
    payment_method:     Optional[str]   = Field("Mailed Check", description="Bank Withdrawal | Credit Card | Mailed Check")

    class Config:
        json_schema_extra = {"example": {
            "tenure": 2, "monthly_charges": 95.0,
            "internet_service": "Fiber Optic", "contract": "Month-to-Month",
            "payment_method": "Mailed Check", "online_security": "No",
            "tech_support": "No", "paperless_billing": "Yes"
        }}


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction:  int
    risk_level:        str


@app.get("/")
def root():
    return {"message": "Churn Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerInput):
    try:
        return predict(customer.dict())
    except FileNotFoundError:
        raise HTTPException(503, detail="Model not found. Run `python src/train.py` first.")
    except Exception as e:
        raise HTTPException(500, detail=str(e))
