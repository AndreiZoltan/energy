# app/main.py
import os
import pandas as pd
import catboost as cb
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys


# import your module straight from the image

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from energy_pred import GicaHackDataLoader

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
APP_TZ      = os.getenv("TZ", "Europe/Chisinau")
def resolve_path(env_key: str, default_rel: str) -> Path:
    """
    If ENV is set and absolute → use it as-is.
    If ENV is set and relative → resolve relative to PROJECT_ROOT.
    If ENV not set          → use default_rel relative to PROJECT_ROOT.
    """
    raw = os.getenv(env_key, default_rel)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

MODEL_PATH = resolve_path("MODEL_PATH", "models/energy_forecast_model.cbm")
DATA_DIR   = resolve_path("DATA_DIR",   "data/GicaHack")
# HISTORY    = resolve_path("HISTORY_PATH", "data/history.parquet")
FUTURE_HRS = 24
# Optional: fail fast with clear errors
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data dir not found: {DATA_DIR}")

app = FastAPI(title="Energy Forecast (energy_pred powered)")

# ---------- Startup: load model + prepare history ----------
loaded_model = cb.CatBoostRegressor()
loaded_model.load_model(MODEL_PATH)

loader = GicaHackDataLoader(DATA_DIR, verbose=True)
loader.load()
# Build the same enriched frame you train with:
clean_df   = loader.preprocess_and_normalize_consumption()
weather_df = loader.add_weather_features(tz=APP_TZ)
df_enriched = loader.create_advanced_features(weather_df)  # training-ready frame

# ---------- Schemas ----------
class PredictIn(BaseModel):
    future_steps: int | None = None
    quantile: float | None = 0.98  # for simple peak flag

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(body: PredictIn):
    future_steps = body.future_steps or FUTURE_HRS

    # Use the SAME helper you already use in notebooks:
    future_df = loader.predict_future(
        model=loaded_model,
        raw_historical_df=df_enriched,
        future_steps=future_steps,
        feature_creation_function=loader.create_advanced_features
    )

    # Peak flag just like your code:
    q = float(body.quantile or 0.98)
    peak_threshold = float(future_df["prediction"].quantile(q))
    future_df["is_peak"] = (future_df["prediction"] > peak_threshold).astype(int)

    return {
        "quantile": q,
        "peak_threshold": round(peak_threshold, 3),
        "timestamps": [ts.isoformat() for ts in future_df.index],
        "prediction": future_df["prediction"].round(3).tolist(),
        "is_peak": future_df["is_peak"].tolist(),
    }
