import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    # Transit speed: faster routes correlate with higher disruption risk
    df['Distance_per_Day'] = df['Distance_km'] / (df['Lead_Time_Days'] + 1)
    # Compound risk: high geopolitical exposure + low carrier reliability
    df['Risk_Carrier_Interaction'] = df['Geopolitical_Risk_Score'] * (1 - df['Carrier_Reliability_Score'])
    df['Heavy_Cargo'] = (df['Weight_MT'] > df['Weight_MT'].median()).astype(int)
    return df


def encode_features(df: pd.DataFrame, encode_cols: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=encode_cols, drop_first=True, dtype=int)


def build_feature_list(df: pd.DataFrame, base_features: list, encode_cols: list) -> list:
    ohe_features = [c for c in df.columns if any(c.startswith(b + '_') for b in encode_cols)]
    return base_features + ohe_features


def prepare_data(
    df: pd.DataFrame,
    features: list,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    # Preserve feature names for SHAP compatibility
    X_train_res = pd.DataFrame(X_train_res, columns=features)
    y_train_res = pd.Series(y_train_res, name=target)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)

    return X_train_res, X_test, y_train_res, y_test, X_train_sc, X_test_sc, scaler
