import pandas as pd
import numpy as np
import pytest
from src.data.preprocessing import engineer_features, encode_features, build_feature_list


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Date': ['2025-01-15', '2024-06-30'],
        'Distance_km': [5000.0, 10000.0],
        'Weight_MT': [100.0, 300.0],
        'Lead_Time_Days': [10.0, 20.0],
        'Geopolitical_Risk_Score': [5.0, 8.0],
        'Carrier_Reliability_Score': [0.8, 0.6],
        'Transport_Mode': ['Air', 'Rail'],
        'Product_Category': ['Electronics', 'Textiles'],
        'Weather_Condition': ['Clear', 'Storm'],
        'Origin_Port': ['Singapore', 'Rotterdam'],
        'Destination_Port': ['Los Angeles', 'Hamburg'],
        'Fuel_Price_Index': [2.5, 3.0],
        'Disruption_Occurred': [0, 1],
    })


def test_engineer_features_columns(sample_df):
    out = engineer_features(sample_df)
    for col in ['Month', 'Quarter', 'Distance_per_Day', 'Risk_Carrier_Interaction', 'Heavy_Cargo']:
        assert col in out.columns, f'Missing engineered column: {col}'


def test_distance_per_day(sample_df):
    out = engineer_features(sample_df)
    expected = 5000.0 / (10.0 + 1)
    assert abs(out.loc[0, 'Distance_per_Day'] - expected) < 1e-6


def test_heavy_cargo_binary(sample_df):
    out = engineer_features(sample_df)
    assert set(out['Heavy_Cargo'].unique()).issubset({0, 1})


def test_encode_features_drops_originals(sample_df):
    encode_cols = ['Transport_Mode', 'Product_Category', 'Weather_Condition',
                   'Origin_Port', 'Destination_Port']
    out = encode_features(sample_df, encode_cols)
    for col in encode_cols:
        assert col not in out.columns


def test_build_feature_list(sample_df):
    encode_cols = ['Transport_Mode']
    df_enc = encode_features(sample_df, encode_cols)
    base = ['Distance_km']
    features = build_feature_list(df_enc, base, encode_cols)
    assert 'Distance_km' in features
    assert any(f.startswith('Transport_Mode_') for f in features)
