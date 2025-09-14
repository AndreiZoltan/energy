# app/main.py
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd
import catboost as cb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- import your module ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from energy_pred import GicaHackDataLoader  # noqa: E402

APP_TZ = os.getenv("TZ", "Europe/Chisinau")

def resolve_path(env_key: str, default_rel: str) -> Path:
    raw = os.getenv(env_key, default_rel)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

MODEL_PATH = resolve_path("MODEL_PATH", "models/energy_forecast_model.cbm")
FUTURE_HRS = int(os.getenv("FUTURE_HOURS", "24"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

app = FastAPI(title="Energy Forecast (energy_pred powered)", version="1.0")

# ---------- load model on startup ----------
loaded_model = cb.CatBoostRegressor()
loaded_model.load_model(str(MODEL_PATH))  # cast to str for compatibility

# ---------- Schemas ----------
class PredictIn(BaseModel):
    data_dir: str = Field(..., description="Path inside the container to folder with GicaHack CSVs")
    future_steps: int = Field(FUTURE_HRS, ge=1, le=24*14, description="Hours to forecast ahead")
    quantile: float = Field(0.98, ge=0.5, le=0.999, description="Quantile for peak flag (e.g., 0.98)")

class PredictOut(BaseModel):
    quantile: float
    peak_threshold: float
    timestamps: list[str]
    prediction: list[float]
    is_peak: list[int]

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    # Resolve and validate data_dir
    data_dir = Path(body.data_dir)
    if not data_dir.is_absolute():
        # allow relative paths (resolve against project root)
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"data_dir not found: {data_dir}")
    if not data_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"data_dir is not a directory: {data_dir}")

    # Load & build the same enriched frame you train with
    try:
        loader = GicaHackDataLoader(str(data_dir), verbose=False)
        loader.load()
        clean_df   = loader.preprocess_and_normalize_consumption()
        weather_df = loader.add_weather_features(tz=APP_TZ)
        df_enriched = loader.create_advanced_features(weather_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to prepare data from {data_dir}: {e}")

    # Forecast using your helper
    try:
        future_df = loader.predict_future(
            model=loaded_model,
            raw_historical_df=df_enriched,
            future_steps=body.future_steps,
            feature_creation_function=loader.create_advanced_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Peak flag (per-request quantile)
    q = float(body.quantile)
    peak_threshold = float(future_df["prediction"].quantile(q))
    future_df["is_peak"] = (future_df["prediction"] > peak_threshold).astype(int)

    return PredictOut(
        quantile=q,
        peak_threshold=round(peak_threshold, 3),
        timestamps=[ts.isoformat() for ts in future_df.index],
        prediction=[float(x) for x in future_df["prediction"].round(3).tolist()],
        is_peak=future_df["is_peak"].tolist(),
    )
