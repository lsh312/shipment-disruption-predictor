import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_model_configs(random_state: int = 42) -> dict:
    return {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=random_state),
            'scaled': True,
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=6, random_state=random_state),
            'scaled': False,
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state),
            'scaled': False,
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=random_state
            ),
            'scaled': False,
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'scaled': True,
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'scaled': True,
        },
    }


def _as_array(X) -> np.ndarray:
    return X.values if isinstance(X, pd.DataFrame) else X


def train_and_evaluate(
    model_configs: dict,
    X_train_res: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_res: pd.Series,
    y_test: pd.Series,
    X_train_sc: np.ndarray,
    X_test_sc: np.ndarray,
    cv,
) -> dict:
    results = {}
    for name, cfg in model_configs.items():
        model = cfg['model']
        X_tr = X_train_sc if cfg['scaled'] else _as_array(X_train_res)
        X_te = X_test_sc if cfg['scaled'] else _as_array(X_test)

        cv_scores = cross_val_score(model, X_tr, y_train_res, cv=cv, scoring='roc_auc')
        model.fit(X_tr, y_train_res)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        results[name] = {
            'model':     model,
            'scaled':    cfg['scaled'],
            'y_pred':    y_pred,
            'y_prob':    y_prob,
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred), 4),
            'recall':    round(recall_score(y_test, y_pred), 4),
            'f1':        round(f1_score(y_test, y_pred), 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
            'cv_mean':   round(cv_scores.mean(), 4),
            'cv_std':    round(cv_scores.std(), 4),
        }
    return results


def tune_logistic_regression(
    X_train_sc: np.ndarray,
    y_train_res: pd.Series,
    cv,
    random_state: int = 42,
    scoring: str = 'recall',
) -> tuple:
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
    }
    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        param_grid, cv=cv, scoring=scoring, n_jobs=-1,
    )
    search.fit(X_train_sc, y_train_res)
    return search.best_estimator_, search.best_params_, round(search.best_score_, 4)
