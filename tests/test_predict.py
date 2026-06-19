import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('/content/mlops-demand-forecasting')

from src.features import (GeohashEncoder, TimeFeatureExtractor,
                          CategoricalEncoder, FEATURE_COLS)

def test_model_loads():
    model = joblib.load('artifacts/lgb_model.pkl')
    assert model is not None
    print("✓ test_model_loads passed")

def test_prediction_is_float():
    model   = joblib.load('artifacts/lgb_model.pkl')
    geo_enc = joblib.load('artifacts/geo_encoder.pkl')
    time_ext= joblib.load('artifacts/time_extractor.pkl')
    cat_enc = joblib.load('artifacts/cat_encoder.pkl')

    df = pd.DataFrame([{
        'geohash6'  : 'qp03wnh',
        'timestamp' : '2024-01-15 08:30:00',
        'road_type' : 'primary',
        'weather'   : 'clear'
    }])
    df = geo_enc.transform(df)
    df = time_ext.transform(df)
    df = cat_enc.transform(df)

    pred = float(model.predict(df[FEATURE_COLS])[0])
    assert isinstance(pred, float)
    assert pred >= 0, f"Negative demand prediction: {pred}"
    assert pred < 100, f"Unreasonably high prediction: {pred}"
    print(f"✓ test_prediction_is_float passed (demand={pred:.4f})")

def test_batch_prediction():
    model    = joblib.load('artifacts/lgb_model.pkl')
    geo_enc  = joblib.load('artifacts/geo_encoder.pkl')
    time_ext = joblib.load('artifacts/time_extractor.pkl')
    cat_enc  = joblib.load('artifacts/cat_encoder.pkl')

    df = pd.DataFrame([
        {'geohash6': 'qp03wnh',  'timestamp': '2024-01-15 08:00:00',
         'road_type': 'primary',     'weather': 'clear'},
        {'geohash6': 'qp03wj0',  'timestamp': '2024-01-15 17:30:00',
         'road_type': 'secondary',   'weather': 'rain'},
        {'geohash6': 'qp03wj1',  'timestamp': '2024-01-15 12:00:00',
         'road_type': 'residential', 'weather': 'cloudy'},
    ])
    df   = geo_enc.transform(df)
    df   = time_ext.transform(df)
    df   = cat_enc.transform(df)
    preds = model.predict(df[FEATURE_COLS])

    assert len(preds) == 3
    assert all(p >= 0 for p in preds)
    print(f"✓ test_batch_prediction passed (preds={[round(p,4) for p in preds]})")

if __name__ == '__main__':
    test_model_loads()
    test_prediction_is_float()
    test_batch_prediction()
    print("\n✅ All prediction tests passed!")
