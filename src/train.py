import os
import warnings
import pandas as pd
import numpy as np
import mlflow
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

FEATURE_COLS = [
    'geohash_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'road_type_encoded', 'weather_encoded'
]
TARGET_COL = 'demand'


def train_lightgbm(X_train, X_val, y_train, y_val, params):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data]
    )

    y_pred = model.predict(X_val)
    rmse   = np.sqrt(mean_squared_error(y_val, y_pred))
    mae    = mean_absolute_error(y_val, y_pred)
    r2     = r2_score(y_val, y_pred)

    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)
    mlflow.log_metric('r2', r2)

    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, 'artifacts/lgb_model.pkl')

    return model, rmse


def main():
    df = pd.read_csv('data/processed/features.csv')
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training LightGBM...")
    with mlflow.start_run(run_name='lgb_baseline'):
        lgb_params = {
            'objective' : 'regression',
            'metric'    : 'rmse',
            'num_leaves': 31
        }
        mlflow.log_params(lgb_params)
        lgb_model, lgb_rmse = train_lightgbm(
            X_train, X_val, y_train, y_val, lgb_params
        )
        print(f"LightGBM RMSE: {lgb_rmse:.4f}")

    print("Model saved to artifacts/lgb_model.pkl")
    return lgb_model


if __name__ == '__main__':
    main()
