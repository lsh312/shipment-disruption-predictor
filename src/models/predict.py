import numpy as np
import pandas as pd
import joblib


def load_model(path: str):
    return joblib.load(path)


def load_scaler(path: str):
    return joblib.load(path)


def predict(model, X: pd.DataFrame, scaler=None) -> dict:
    X_input = scaler.transform(X) if scaler is not None else X.values
    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]
    return {'predictions': y_pred, 'probabilities': y_prob}
