"""
FastAPI service for the hotel booking cancellation prediction model.

This matches the v4 notebook: 28 model features (29 minus room_mismatch,
see note below), and a K-Prototypes clustering step using 13 of those
features.

`room_mismatch` (whether the assigned room differs from the reserved
room) was engineered in the notebook and is a strong bivariate predictor,
but it is excluded from the model here. It is only knowable once a
booking has already survived to the hotel's room allocation stage close
to arrival, so it is a form of target leakage: its value is a near
consequence of the booking not having cancelled, not an independent
signal available at prediction time. It remains a legitimate and useful
finding in the notebook's EDA, just not a fair input to this model.

Run locally with:
    uvicorn app:app --reload --port 8000

Deployed on Render using the start command:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import json
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'model_artifacts')

# ---------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------
try:
    preprocessor = joblib.load(f'{ARTIFACT_DIR}/preprocessor.joblib')
    rf_model = joblib.load(f'{ARTIFACT_DIR}/rf_model.joblib')
    kproto_model = joblib.load(f'{ARTIFACT_DIR}/kproto_model.joblib')
    cluster_scaler = joblib.load(f'{ARTIFACT_DIR}/cluster_scaler.joblib')
    with open(f'{ARTIFACT_DIR}/metadata.json') as f:
        METADATA = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        "Model artifacts not found. Run the export cells at the end of the "
        "notebook and copy the 'model_artifacts' folder next to this file "
        f"before starting the API. Original error: {e}"
    )

FEATURE_ORDER = METADATA['feature_order']
KPROTO_NUMERIC = METADATA['kproto_numeric']
KPROTO_CATEGORICAL = METADATA['kproto_categorical']
CATEGORICAL_INDICES = METADATA['categorical_indices']

# Risk tier thresholds on predicted cancellation probability.
# Confirm these against the validation-set probability distribution
# for your own trained model before relying on them operationally.
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60

app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description="Predicts the probability that a booking will be cancelled, "
                "and assigns the booking to a customer persona cluster.",
    version="2.0.0",
)


# ---------------------------------------------------------------------
# Request schema
#
# Fields are grouped by how much they matter to the prediction, based on
# the feature importance and Cramer's V results from the v4 notebook, and
# by whether a support rep could realistically know the value at the time
# of the call. Core fields are required. Advanced fields have defaults
# and can be left out entirely.
# ---------------------------------------------------------------------
class BookingProfile(BaseModel):
    # --- Core fields: the strongest predictors, and the ones a rep would
    #     realistically know when reviewing an upcoming booking ---
    hotel: str = Field(..., description="'City Hotel' or 'Resort Hotel'")
    lead_time: int = Field(..., ge=0, description="Days between booking and arrival")
    adr: float = Field(..., ge=0, description="Average daily rate")
    total_nights: int = Field(..., ge=1)
    adults: int = Field(..., ge=1)
    children: int = Field(0, ge=0)
    market_segment: str = Field(..., description="e.g. 'Online TA', 'Direct', 'Corporate'")
    customer_type: str = Field(..., description="e.g. 'Transient', 'Contract', 'Group'")
    deposit_type: str = Field(..., description="'No Deposit', 'Non Refund', or 'Refundable'")
    previous_cancellations: int = Field(0, ge=0, description="Strongest behavioural signal after lead time")
    is_repeated_guest: int = Field(0, ge=0, le=1, description="1 if this guest has stayed before")
    booking_changes: int = Field(0, ge=0)
    total_of_special_requests: int = Field(0, ge=0)
    required_car_parking_spaces: int = Field(0, ge=0)

    # --- Advanced fields: shown to matter less in the notebook's Cramer's V
    #     / feature importance results (meal, distribution_channel, country,
    #     previous_bookings_not_canceled). Note that room_mismatch is not
    #     listed here at all: it is excluded from the model entirely, see
    #     the module docstring above. ---
    arrival_date_year: int = Field(default_factory=lambda: date.today().year)
    arrival_date_month: str = Field(default_factory=lambda: date.today().strftime('%B'))
    arrival_date_week_number: int = Field(default_factory=lambda: date.today().isocalendar()[1])
    arrival_date_day_of_month: int = Field(default_factory=lambda: date.today().day)
    stays_in_weekend_nights: int = Field(0, ge=0)
    stays_in_week_nights: int = Field(None, ge=0)
    babies: int = Field(0, ge=0)
    meal: str = Field("BB", description="Lowest Cramer's V of the tested categorical features")
    country: str = Field("PRT", description="Not individually tested in the v4 notebook; high cardinality")
    distribution_channel: str = Field("Direct", description="Overlaps conceptually with market_segment")
    previous_bookings_not_canceled: int = Field(0, ge=0)
    agent: int = Field(0, ge=0)
    days_in_waiting_list: int = Field(0, ge=0)

    def to_feature_row(self) -> dict:
        data = self.dict()
        if data['stays_in_week_nights'] is None:
            data['stays_in_week_nights'] = data['total_nights'] - data['stays_in_weekend_nights']
        return data


class PredictionResponse(BaseModel):
    cancellation_probability: float
    risk_tier: str
    customer_persona_cluster: str


# ---------------------------------------------------------------------
# Helper: assemble a single feature row in the exact column order and
# dtype layout the trained pipeline expects
# ---------------------------------------------------------------------
def build_feature_row(profile: BookingProfile) -> pd.DataFrame:
    raw = profile.to_feature_row()
    row = pd.DataFrame([raw])

    # --- Step 1: assign the customer persona cluster using K-Prototypes ---
    cluster_input = row[KPROTO_NUMERIC].copy()
    cluster_input[KPROTO_NUMERIC] = cluster_scaler.transform(cluster_input[KPROTO_NUMERIC])
    for col in KPROTO_CATEGORICAL:
        cluster_input[col] = row[col].values
    cluster_matrix = cluster_input[KPROTO_NUMERIC + KPROTO_CATEGORICAL].values
    persona_cluster = kproto_model.predict(cluster_matrix, categorical=CATEGORICAL_INDICES)[0]
    row['customer_persona_cluster'] = str(persona_cluster)

    # --- Step 2: reorder columns to match training-time feature order ---
    missing = [c for c in FEATURE_ORDER if c not in row.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing engineered columns: {missing}")

    return row[FEATURE_ORDER]


def risk_tier_from_probability(p: float) -> str:
    if p < LOW_RISK_MAX:
        return "Low"
    elif p < MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Simple liveness check used by Render and by n8n before calling /predict."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: BookingProfile):
    """
    Accepts a booking profile and returns the predicted cancellation
    probability, a risk tier derived from that probability, and the
    customer persona cluster assigned by the K-Prototypes model.
    """
    try:
        feature_row = build_feature_row(profile)
        encoded_row = preprocessor.transform(feature_row)
        probability = float(rf_model.predict_proba(encoded_row)[0, 1])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        cancellation_probability=round(probability, 4),
        risk_tier=risk_tier_from_probability(probability),
        customer_persona_cluster=str(feature_row['customer_persona_cluster'].iloc[0]),
    )


@app.get("/")
def root():
    return {
        "message": "Hotel Booking Cancellation Prediction API",
        "endpoints": {
            "GET /health": "Liveness check",
            "POST /predict": "Predict cancellation probability from a booking profile",
        },
    }
