import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (GeohashEncoder, TimeFeatureExtractor,
                          CategoricalEncoder, FEATURE_COLS)

def test_feature_pipeline_runs():
    """Test full feature pipeline runs without errors"""
    df = pd.DataFrame([{
        'geohash6'  : 'qp03wnh',
        'timestamp' : '2024-01-15 08:30:00',
        'road_type' : 'primary',
        'weather'   : 'clear',
        'demand'    : 2.5
    }])
    geo  = GeohashEncoder()
    time = TimeFeatureExtractor()
    cat  = CategoricalEncoder()
    df   = geo.fit_transform(df)
    df   = time.fit_transform(df)
    df   = cat.fit_transform(df)
    for col in FEATURE_COLS:
        assert col in df.columns

def test_batch_feature_pipeline():
    """Test pipeline handles multiple rows"""
    df = pd.DataFrame([
        {'geohash6': 'qp03wnh',  'timestamp': '2024-01-15 08:00:00',
         'road_type': 'primary',     'weather': 'clear',  'demand': 2.5},
        {'geohash6': 'qp03wj0',  'timestamp': '2024-01-15 17:30:00',
         'road_type': 'secondary',   'weather': 'rain',   'demand': 3.1},
        {'geohash6': 'qp03wj1',  'timestamp': '2024-01-15 12:00:00',
         'road_type': 'residential', 'weather': 'cloudy', 'demand': 1.9},
    ])
    geo  = GeohashEncoder()
    time = TimeFeatureExtractor()
    cat  = CategoricalEncoder()
    df   = geo.fit_transform(df)
    df   = time.fit_transform(df)
    df   = cat.fit_transform(df)
    assert len(df) == 3
    for col in FEATURE_COLS:
        assert col in df.columns
