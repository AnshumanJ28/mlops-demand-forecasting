import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (GeohashEncoder, TimeFeatureExtractor,
                          CategoricalEncoder, FEATURE_COLS, TARGET_COL)

def test_geohash_encoder():
    df = pd.DataFrame({
        'geohash6'  : ['qp03wnh', 'qp03wj0', 'qp03wnh'],
        'timestamp' : ['2024-01-01 08:00:00'] * 3,
        'road_type' : ['primary'] * 3,
        'weather'   : ['clear'] * 3,
        'demand'    : [2.5, 3.1, 2.8]
    })
    enc = GeohashEncoder()
    out = enc.fit_transform(df)
    assert 'geohash_encoded' in out.columns
    assert out['geohash_encoded'].isnull().sum() == 0

def test_time_feature_extractor():
    df = pd.DataFrame({
        'geohash6'  : ['qp03wnh'] * 3,
        'timestamp' : ['2024-01-01 08:00:00',
                       '2024-01-01 14:00:00',
                       '2024-01-06 10:00:00'],
        'road_type' : ['primary'] * 3,
        'weather'   : ['clear'] * 3,
        'demand'    : [2.5, 3.1, 2.8]
    })
    ext = TimeFeatureExtractor()
    out = ext.fit_transform(df)
    assert 'hour' in out.columns
    assert 'is_weekend' in out.columns
    assert 'is_rush_hour' in out.columns
    assert 'hour_sin' in out.columns
    assert 'hour_cos' in out.columns
    assert out.loc[0, 'is_rush_hour'] == 1
    assert out.loc[1, 'is_rush_hour'] == 0
    assert out.loc[2, 'is_weekend']   == 1

def test_categorical_encoder():
    df = pd.DataFrame({
        'geohash6'  : ['qp03wnh'] * 4,
        'timestamp' : ['2024-01-01 08:00:00'] * 4,
        'road_type' : ['primary', 'secondary', 'residential', 'motorway'],
        'weather'   : ['clear', 'rain', 'fog', 'cloudy'],
        'demand'    : [2.5, 3.1, 2.8, 1.9]
    })
    enc = CategoricalEncoder()
    out = enc.fit_transform(df)
    assert 'road_type_encoded' in out.columns
    assert 'weather_encoded' in out.columns
    assert out['road_type_encoded'].isnull().sum() == 0

def test_feature_cols_complete():
    df = pd.DataFrame({
        'geohash6'  : ['qp03wnh'] * 5,
        'timestamp' : ['2024-01-01 08:00:00'] * 5,
        'road_type' : ['primary'] * 5,
        'weather'   : ['clear'] * 5,
        'demand'    : [2.5, 3.1, 2.8, 1.9, 4.2]
    })
    geo  = GeohashEncoder()
    time = TimeFeatureExtractor()
    cat  = CategoricalEncoder()
    df   = geo.fit_transform(df)
    df   = time.fit_transform(df)
    df   = cat.fit_transform(df)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"
