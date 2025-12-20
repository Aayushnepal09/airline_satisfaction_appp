# api/app.py
"""FastAPI service for Airline Passenger Satisfaction prediction.

Loads a pre-trained binary classification pipeline (joblib) and exposes:
- GET /health
- POST /predict

Target: satisfaction_binary (1 = satisfied, 0 = neutral/dissatisfied)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure joblib unpickling can find the pipeline module
# (we keep the module name stable)
from housing_pipeline import CATEGORICAL_COLS, NUMERIC_COLS

# Load .env from project root (works whether you run from root or api/)
try:
    from dotenv import load_dotenv
except Exception as e:
    raise RuntimeError(
        "python-dotenv is required. Install it in api/requirements.txt: pip install python-dotenv"
    ) from e

# -----------------------------------------------------------------------------
# Paths & environment
# -----------------------------------------------------------------------------
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent  # one level above /api
ENV_PATH = PROJECT_ROOT / ".env"

# Load env vars if .env exists
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Not fatal; docker may set env vars via compose
    pass


def _resolve_path(p: str | Path, base: Path) -> Path:
    """Resolve p relative to base if not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def _choose_model_path() -> Path:
    """Pick a model path that exists, preferring env var then common defaults."""
    env_val = os.getenv("MODEL_PATH")

    candidates: List[Path] = []

    # 1) If env set, resolve relative to project root
    if env_val:
        candidates.append(_resolve_path(env_val, PROJECT_ROOT))

    # 2) Local defaults
    candidates.append(PROJECT_ROOT / "models" / "global_best_model_optuna.pkl")
    candidates.append(PROJECT_ROOT / "models" / "global_best_model.pkl")

    # 3) Docker defaults
    candidates.append(Path("/app/models/global_best_model_optuna.pkl"))
    candidates.append(Path("/app/models/global_best_model.pkl"))

    for c in candidates:
        if c.exists():
            return c.resolve()

    # If none exist, return the first candidate for better error message
    return candidates[0].resolve() if candidates else (PROJECT_ROOT / "models").resolve()


MODEL_PATH = _choose_model_path()
FEATURE_COLUMNS = list(NUMERIC_COLS) + list(CATEGORICAL_COLS)

CLASS_LABELS = {
    0: "neutral or dissatisfied",
    1: "satisfied",
}

app = FastAPI(
    title="Airline Passenger Satisfaction Prediction API",
    description="FastAPI service for predicting passenger satisfaction (binary classification)",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Checked .env at: {ENV_PATH} (exists={ENV_PATH.exists()})\n"
            f"Tip (local): set MODEL_PATH=models/global_best_model_optuna.pkl in .env\n"
            f"Tip (docker): set MODEL_PATH=/app/models/global_best_model_optuna.pkl"
        )

    print(f"✅ Loading model from: {path}")
    m = joblib.load(path)
    print("✅ Model loaded successfully!")
    print(f"   Model type: {type(m).__name__}")
    if hasattr(m, "named_steps"):
        print(f"   Pipeline steps: {list(m.named_steps.keys())}")
    return m


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ ERROR: Failed to load model from {MODEL_PATH}")
    print(f"   Error: {e}")
    raise RuntimeError(f"Failed to load model: {e}")


# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "gender": "Female",
                        "customer_type": "Loyal Customer",
                        "age": 35,
                        "type_of_travel": "Business travel",
                        "travel_class": "Business",
                        "flight_distance": 1200,
                        "inflight_wifi_service": 4,
                        "departure_arrival_time_convenient": 3,
                        "ease_of_online_booking": 3,
                        "gate_location": 2,
                        "food_and_drink": 3,
                        "online_boarding": 4,
                        "seat_comfort": 4,
                        "inflight_entertainment": 4,
                        "on_board_service": 4,
                        "leg_room_service": 4,
                        "baggage_handling": 4,
                        "checkin_service": 4,
                        "inflight_service": 4,
                        "cleanliness": 4,
                        "departure_delay_minutes": 0,
                        "arrival_delay_minutes": 0,
                    }
                ]
            }
        }


class PredictResponse(BaseModel):
    predictions: List[int]
    labels: List[str]
    probabilities: Optional[List[float]] = None
    count: int


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Airline Passenger Satisfaction Prediction API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_loaded": str(model is not None),
        "model_path": str(MODEL_PATH),
        "env_loaded": str(ENV_PATH.exists()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided.")

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not convert input to DataFrame: {e}")

    missing = sorted(set(FEATURE_COLUMNS) - set(X.columns))
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    X = X[FEATURE_COLUMNS]

    try:
        preds = model.predict(X)
        preds_list = [int(p) for p in preds]
        labels = [CLASS_LABELS.get(p, str(p)) for p in preds_list]

        proba_list = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba_list = [float(p[1]) for p in proba]  # positive class probability

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return PredictResponse(
        predictions=preds_list,
        labels=labels,
        probabilities=proba_list,
        count=len(preds_list),
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Airline Passenger Satisfaction Prediction API - Starting Up")
    print("=" * 80)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f".env path: {ENV_PATH} (exists={ENV_PATH.exists()})")
    print(f"Resolved MODEL_PATH: {MODEL_PATH} (exists={MODEL_PATH.exists()})")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests!")
    print("=" * 80 + "\n")
