import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import GeohashEncoder, TimeFeatureExtractor, CategoricalEncoder, FEATURE_COLS

app = FastAPI(title="Traffic Demand Forecasting API", version="1.0")

model    = joblib.load('artifacts/lgb_model.pkl')
geo_enc  = joblib.load('artifacts/geo_encoder.pkl')
time_ext = joblib.load('artifacts/time_extractor.pkl')
cat_enc  = joblib.load('artifacts/cat_encoder.pkl')

prediction_log = []


class PredictRequest(BaseModel):
    geohash6 : str
    timestamp: str
    road_type: str
    weather  : str


class PredictResponse(BaseModel):
    demand       : float
    model_version: str = "lgb_baseline"
    timestamp    : str


@app.get("/")
def root():
    return {"status": "ok", "message": "Traffic Demand Forecasting API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "lgb_baseline"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame([{
        'geohash6' : request.geohash6,
        'timestamp': request.timestamp,
        'road_type': request.road_type,
        'weather'  : request.weather
    }])

    df     = geo_enc.transform(df)
    df     = time_ext.transform(df)
    df     = cat_enc.transform(df)
    demand = float(model.predict(df[FEATURE_COLS].values)[0])

    prediction_log.append({
        'timestamp'  : datetime.now().isoformat(),
        'geohash6'   : request.geohash6,
        'road_type'  : request.road_type,
        'weather'    : request.weather,
        'demand_pred': demand
    })

    return PredictResponse(
        demand=round(demand, 4),
        timestamp=datetime.now().isoformat()
    )


@app.get("/logs")
def get_logs():
    return {"total_predictions": len(prediction_log), "logs": prediction_log[-10:]}
