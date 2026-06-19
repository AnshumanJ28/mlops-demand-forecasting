import os
import json
import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

FEATURE_COLS = [
    'geohash_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'road_type_encoded', 'weather_encoded'
]
TARGET_COL         = 'demand'
DRIFT_THRESHOLD    = 0.09
COL_DRIFT_THRESHOLD= 0.1


def load_reference_data():
    df = pd.read_csv('data/processed/features.csv')
    return df[FEATURE_COLS + [TARGET_COL]]


def simulate_production_data(reference_df, drift=False):
    df = reference_df.sample(n=1000, random_state=99).copy()
    if drift:
        df['hour']        = (df['hour'] + 6) % 24
        df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_rush_hour']= df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df[TARGET_COL]    = df[TARGET_COL] * 1.8 + np.random.normal(0, 0.5, len(df))
    else:
        df[TARGET_COL] = df[TARGET_COL] + np.random.normal(0, 0.05, len(df))
    return df


def run_monitor(drift=False):
    reference_df  = load_reference_data()
    production_df = simulate_production_data(reference_df, drift=drift)

    schema       = DataDefinition(numerical_columns=FEATURE_COLS + [TARGET_COL])
    ref_dataset  = Dataset.from_pandas(reference_df,  data_definition=schema)
    prod_dataset = Dataset.from_pandas(production_df, data_definition=schema)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_dataset, current_data=prod_dataset)

    result_dict = result.dict()
    os.makedirs('reports', exist_ok=True)
    with open('reports/drift_report.json', 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    col_scores  = {}
    drift_count = 0
    drift_share = 0.0

    for metric in result_dict.get('metrics', []):
        name = metric.get('metric_name', '')
        val  = metric.get('value', 0)

        if 'DriftedColumnsCount' in name and isinstance(val, dict):
            drift_count = val.get('count', 0)
            drift_share = val.get('share', 0.0)

        if 'ValueDrift' in name and isinstance(val, float):
            col = metric.get('config', {}).get('column', '')
            if col:
                col_scores[col] = val

    drift_detected = drift_share > DRIFT_THRESHOLD

    print(f"Drifted features : {int(drift_count)} / {len(col_scores)}")
    print(f"Drift share      : {drift_share:.2%}")
    print(f"Drift detected   : {drift_detected}")

    for col, score in col_scores.items():
        status = "DRIFTED" if score > COL_DRIFT_THRESHOLD else "ok"
        print(f"  {col:<25} {score:.4f}  {status}")

    if drift_detected:
        print("ALERT: Data drift detected. Retraining recommended.")
    else:
        print("No significant drift detected.")

    return drift_detected, drift_share
