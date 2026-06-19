# mlops-demand-forecasting

An end-to-end MLOps pipeline for traffic demand forecasting, covering data versioning, feature engineering, model training with experiment tracking, REST API serving, drift monitoring, and automated CI/CD.

---

## Stack

| Layer               | Tool                             |
|---------------------|----------------------------------|
| Data versioning     | DVC + Google Drive               |
| Feature engineering | scikit-learn custom transformers |
| Model training      | LightGBM                         |
| Experiment tracking | MLflow + DagsHub                 |
| Model serving       | FastAPI + Uvicorn                |
| Drift monitoring    | Evidently AI                     |
| CI/CD               | GitHub Actions                   |

---

## Project Structure

    mlops-demand-forecasting/
    ├── src/
    │   ├── features.py       # feature engineering pipeline
    │   ├── train.py          # model training and mlflow logging
    │   └── monitor.py        # evidently drift monitoring
    ├── api/
    │   └── app.py            # fastapi prediction endpoint
    ├── tests/
    │   ├── test_features.py  # feature pipeline unit tests
    │   └── test_predict.py   # prediction pipeline unit tests
    ├── data/
    │   ├── raw/              # dvc-tracked raw data
    │   └── processed/        # dvc-tracked processed features
    ├── artifacts/            # saved model and encoders
    ├── reports/              # drift reports (json)
    ├── .github/workflows/
    │   └── ci.yml            # github actions ci pipeline
    └── requirements.txt

---

## Setup

**Prerequisites:** Python 3.10+, Google Drive account, DagsHub account

    git clone https://github.com/AnshumanJ28/mlops-demand-forecasting.git
    cd mlops-demand-forecasting
    pip install -r requirements.txt
    dvc pull

---

## Running the Pipeline

**1. Train model**

    python src/train.py

Logs parameters, RMSE, MAE, and R2 to MLflow. Model saved to artifacts/lgb_model.pkl.

**2. Start API**

    uvicorn api.app:app --host 0.0.0.0 --port 8000

**3. Predict**

    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"geohash6": "qp03wnh", "timestamp": "2024-01-15 08:30:00", "road_type": "primary", "weather": "clear"}'

Response:

    {"demand": 4.3063, "model_version": "lgb_baseline", "timestamp": "2024-01-15T08:30:00"}

**4. Run drift monitoring**

    python -c "from src.monitor import run_monitor; run_monitor()"

**5. Run tests**

    pytest tests/ -v

---

## Experiment Tracking

MLflow experiments tracked on DagsHub:
https://dagshub.com/a.pandeyj28/mlops-demand-forecasting/experiments

---

## CI/CD

GitHub Actions runs on every push to main:

1. Install dependencies
2. Run feature pipeline tests
3. Run prediction pipeline tests
4. Verify monitor and train modules load correctly

Status: ![CI](https://github.com/AnshumanJ28/mlops-demand-forecasting/actions/workflows/ci.yml/badge.svg)

---

## Dataset

Synthetic traffic demand dataset with 50,000 rows mimicking real spatio-temporal demand patterns.

| Column      | Description                                  |
|-------------|----------------------------------------------|
| geohash6    | Location encoded as geohash                  |
| timestamp   | 30-minute interval datetime                  |
| road_type   | primary / secondary / residential / motorway |
| weather     | clear / rain / fog / cloudy                  |
| demand      | Target variable (continuous)                 |

---

## Features Engineered

| Feature           | Description                          |
|-------------------|--------------------------------------|
| geohash_encoded   | Integer encoding of geohash location |
| hour              | Hour extracted from timestamp        |
| day_of_week       | Day of week                          |
| is_weekend        | Binary flag for weekend              |
| is_rush_hour      | Binary flag for peak hours (7-9, 17-19) |
| hour_sin          | Cyclical sine encoding of hour       |
| hour_cos          | Cyclical cosine encoding of hour     |
| road_type_encoded | Label encoded road type              |
| weather_encoded   | Label encoded weather condition      |
