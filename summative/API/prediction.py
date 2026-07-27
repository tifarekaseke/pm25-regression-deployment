"""FastAPI service for PM2.5 prediction and protected model retraining."""

from __future__ import annotations

import hmac
import json
import os
import threading
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from summative.linear_regression.training import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    train_models_from_dataframe,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
UPLOAD_DIR = ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WindDirection = Literal[
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
Station = Literal[
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "month": 7,
                "hour": 14,
                "pm10": 120.0,
                "so2": 12.0,
                "no2": 45.0,
                "co": 900.0,
                "o3": 70.0,
                "temperature": 28.5,
                "pressure": 1004.0,
                "dew_point": 17.2,
                "rainfall": 0.0,
                "wind_speed": 2.4,
                "wind_direction": "SE",
                "station": "Aotizhongxin",
            }
        }
    )

    month: int = Field(ge=1, le=12, description="Month number")
    hour: int = Field(ge=0, le=23, description="Hour of day")
    pm10: float = Field(ge=0, le=1000, description="PM10 concentration in µg/m³")
    so2: float = Field(ge=0, le=500, description="SO2 concentration in µg/m³")
    no2: float = Field(ge=0, le=500, description="NO2 concentration in µg/m³")
    co: float = Field(ge=0, le=10000, description="CO concentration in µg/m³")
    o3: float = Field(ge=0, le=500, description="O3 concentration in µg/m³")
    temperature: float = Field(ge=-40, le=50, description="Temperature in °C")
    pressure: float = Field(ge=900, le=1100, description="Atmospheric pressure in hPa")
    dew_point: float = Field(ge=-50, le=40, description="Dew point in °C")
    rainfall: float = Field(ge=0, le=100, description="Hourly rainfall in mm")
    wind_speed: float = Field(ge=0, le=50, description="Wind speed in m/s")
    wind_direction: WindDirection
    station: Station

    def to_dataframe(self) -> pd.DataFrame:
        row = {
            "month": self.month,
            "hour": self.hour,
            "PM10": self.pm10,
            "SO2": self.so2,
            "NO2": self.no2,
            "CO": self.co,
            "O3": self.o3,
            "TEMP": self.temperature,
            "PRES": self.pressure,
            "DEWP": self.dew_point,
            "RAIN": self.rainfall,
            "WSPM": self.wind_speed,
            "wd": self.wind_direction,
            "station": self.station,
        }
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)


class PredictionResponse(BaseModel):
    predicted_pm25: float
    unit: str
    model: str


app = FastAPI(
    title="PM2.5 Prediction API",
    description=(
        "Predict hourly PM2.5 concentration and retrain the regression model "
        "from an uploaded labeled CSV."
    ),
    version="1.0.0",
)

DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]
configured_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Retrain-Token"],
)

_model_lock = threading.RLock()
_model = None
_metadata: dict = {}


def load_model() -> None:
    global _model, _metadata
    if not MODEL_PATH.exists():
        _model = None
        _metadata = {}
        return
    with _model_lock:
        _model = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        else:
            _metadata = {"best_model_name": "Saved regression model"}


load_model()


@app.get("/")
def root() -> dict:
    return {
        "message": "PM2.5 prediction API is running",
        "docs": "/docs",
        "model_loaded": _model is not None,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy" if _model is not None else "model_missing",
        "model_loaded": _model is not None,
        "model": _metadata.get("best_model_name"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model file is missing. Run the notebook to generate "
                "models/best_model.joblib before starting or deploying the API."
            ),
        )

    with _model_lock:
        predicted_value = float(_model.predict(payload.to_dataframe())[0])

    return PredictionResponse(
        predicted_pm25=round(max(0.0, predicted_value), 2),
        unit="µg/m³",
        model=_metadata.get("best_model_name", "Best saved regression model"),
    )


@app.post("/retrain")
async def retrain(
    file: UploadFile = File(..., description="CSV with model features and PM2.5 target"),
    x_retrain_token: str = Header(..., alias="X-Retrain-Token"),
) -> dict:
    expected_token = os.getenv("RETRAIN_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RETRAIN_TOKEN is not configured on the server.",
        )
    if not hmac.compare_digest(x_retrain_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid retraining token.",
        )
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload must be a CSV file.")

    upload_path = UPLOAD_DIR / "latest_retraining_data.csv"
    content = await file.read()
    max_bytes = 50 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="CSV exceeds the 50 MB upload limit.")
    upload_path.write_bytes(content)

    try:
        frame = pd.read_csv(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {missing}",
        )

    try:
        result = train_models_from_dataframe(
            frame,
            model_dir=ROOT / "models",
            sample_size=int(os.getenv("RETRAIN_MAX_ROWS", "50000")),
        )
        load_model()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc

    return {
        "message": "Retraining completed and the API model was reloaded.",
        "best_model": result.best_model_name,
        "training_rows": result.sample_size,
        "metrics": result.metrics,
    }
