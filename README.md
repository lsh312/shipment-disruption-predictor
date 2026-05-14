# Supply Chain Disruption Prediction

Binary classification model to predict whether a shipment will experience a disruption.

**Course:** Machine Learning II — MBDS 2026  
**Team:** Jan · Caspar · Lea · Ghezlan · Fouad  
**Best model:** Logistic Regression (tuned) — ROC-AUC 0.818, Recall 0.770

---

## Repository structure

```
├── data/raw/                   # Raw dataset (CSV)
├── notebooks/                  # Exploratory notebooks (kept as-is)
├── reports/                    # Summary report, variable descriptions, presentation
├── src/
│   ├── data/preprocessing.py   # Load → feature engineering → OHE → SMOTE → scale
│   ├── models/train.py         # Model definitions, training loop, hyperparameter tuning
│   ├── models/evaluate.py      # Metrics, ROC curves, SHAP, learning curves
│   ├── models/predict.py       # Inference on new shipments
│   └── visualization/plots.py  # EDA plots
├── models/                     # Saved .joblib artifacts (git-ignored)
├── configs/config.yaml         # All hyperparameters and paths
├── tests/                      # Unit tests (pytest)
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the full pipeline

```bash
python main.py train
```

This will:
1. Load and preprocess `data/raw/global_supply_chain_risk_2026.csv`
2. Apply feature engineering + OHE + SMOTE + StandardScaler
3. Train 6 classifiers with 5-fold cross-validation
4. Tune Logistic Regression with GridSearchCV (optimising recall)
5. Save `models/best_model.joblib` and `models/scaler.joblib`
6. Display evaluation plots (ROC curves, confusion matrices, SHAP, learning curve)

### Score new shipments

```bash
python main.py predict --input path/to/new_shipments.csv
```

Input CSV must match the raw data schema (same columns as the training set).  
Output: `Shipment_ID`, `Predicted_Disruption` (0/1), `Disruption_Probability`.

### Run EDA plots only

```bash
python main.py eda
```

---

## Dataset

`global_supply_chain_risk_2026.csv` — 5,000 shipment records × 14 features.

| Feature | Type | Description |
|---|---|---|
| `Disruption_Occurred` | int (target) | 1 = disruption, 0 = no disruption |
| `Distance_km` | float | Shipment route distance |
| `Weight_MT` | float | Cargo weight (metric tonnes) |
| `Fuel_Price_Index` | float | Fuel cost index at departure |
| `Geopolitical_Risk_Score` | float | Route geopolitical risk (0–10) |
| `Carrier_Reliability_Score` | float | Carrier historical reliability (0–1) |
| `Lead_Time_Days` | float | Scheduled transit time |
| `Transport_Mode` | categorical | Air / Rail / Road / Sea |
| `Product_Category` | categorical | Electronics, Perishables, etc. |
| `Weather_Condition` | categorical | Clear / Fog / Rain / Storm / Hurricane |
| `Origin_Port` | categorical | 9 global ports |
| `Destination_Port` | categorical | 9 global ports |

**Engineered features** (created in `preprocessing.py`):

| Feature | Formula |
|---|---|
| `Distance_per_Day` | `Distance_km / (Lead_Time_Days + 1)` |
| `Risk_Carrier_Interaction` | `Geopolitical_Risk_Score × (1 − Carrier_Reliability_Score)` |
| `Heavy_Cargo` | 1 if `Weight_MT` > median, else 0 |
| `Month`, `Quarter` | Extracted from `Date` |

---

## Model results (base models)

| Model | Accuracy | Recall | F1 | ROC-AUC | CV AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.741 | **0.770** | **0.785** | **0.818** | 0.878 |
| Decision Tree | 0.718 | 0.715 | 0.756 | 0.797 | 0.831 |
| Random Forest | 0.730 | 0.682 | 0.756 | 0.824 | 0.865 |
| Gradient Boosting | 0.721 | 0.742 | 0.765 | 0.804 | 0.867 |
| KNN | 0.658 | 0.666 | 0.705 | 0.721 | 0.814 |
| Naive Bayes | 0.604 | 0.362 | 0.529 | 0.799 | 0.844 |

Logistic Regression was selected as the final model and tuned with GridSearchCV (optimising recall).  
Top features: `Weather_Condition_Hurricane` (25.3%), `Weather_Condition_Storm` (8.0%), `Geopolitical_Risk_Score` (6.7%).

---

## Running tests

```bash
pip install pytest
pytest tests/
```
