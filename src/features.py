import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class GeohashEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.geohash_map = {}

    def fit(self, X, y=None):
        unique = X['geohash6'].unique()
        self.geohash_map = {g: i for i, g in enumerate(unique)}
        return self

    def transform(self, X):
        X = X.copy()
        X['geohash_encoded'] = X['geohash6'].map(self.geohash_map).fillna(-1)
        return X

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['timestamp'] = pd.to_datetime(X['timestamp'])
        X['hour']           = X['timestamp'].dt.hour
        X['day_of_week']    = X['timestamp'].dt.dayofweek
        X['is_weekend']     = (X['day_of_week'] >= 5).astype(int)
        X['is_rush_hour']   = X['hour'].isin([7,8,9,17,18,19]).astype(int)
        X['hour_sin']       = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos']       = np.cos(2 * np.pi * X['hour'] / 24)
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in ['road_type', 'weather']:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col + '_encoded'] = le.transform(X[col].astype(str))
        return X

def build_features(df):
    df = GeohashEncoder().fit_transform(df) if 'geohash6' in df.columns else df
    df = TimeFeatureExtractor().fit_transform(df)
    df = CategoricalEncoder().fit_transform(df)
    return df

FEATURE_COLS = [
    'geohash_encoded',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos',
    'road_type_encoded', 'weather_encoded'
]
TARGET_COL = 'demand'
