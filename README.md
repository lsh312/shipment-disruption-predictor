# Supply Chain Disruption Prediction

Binary classification model to predict whether a shipment will experience a disruption, built for the Global Supply Chain Risk 2026 dataset.

**Course:** Machine Learning II — MBDS 2026  
**Team:** Jan · Caspar · Lea · Ghezlan · Fouad  
**Best model:** Logistic Regression (tuned) — ROC-AUC 0.818, Recall 0.770

---

## About this repository

This project was originally developed as a group assignment for the Machine Learning II course (MBDS 2026). The work was done in Jupyter notebooks — covering EDA, feature engineering, model training, hyperparameter tuning, and interpretability — and delivered as a written report and presentation.

After the course, the project was converted into a modularized Python repository following MLOps best practices: the notebook logic was extracted into reusable `src/` modules, all hyperparameters were moved to a config file, a CLI entry point was added, and an automated HTML report was introduced. The original exploratory notebook is preserved in `notebooks/` alongside the reports and presentation in `reports/`.

The goal of this restructuring is to make the project reproducible, easy to run end-to-end, and extensible toward a real production system — see the [Production roadmap](#production-roadmap) section for what that would look like.

---

## Repository structure

```
├── data/raw/                        # Dataset (downloaded from Kaggle or already present)
├── notebooks/                       # Exploratory notebook (kept as-is)
├── reports/                         # Summary report, variable descriptions, presentation
├── src/
│   ├── data/
│   │   ├── ingestion.py             # Kaggle download with local fallback
│   │   └── preprocessing.py        # Feature engineering → OHE → SMOTE → scale
│   ├── models/
│   │   ├── train.py                 # 6 classifiers + cross-validation + tuning
│   │   ├── evaluate.py              # Metrics, ROC curves, SHAP, learning curves
│   │   └── predict.py              # Inference on new shipments
│   ├── visualization/plots.py       # EDA plots
│   └── reporting/report.py         # Self-contained HTML report generator
├── output/                          # Generated (git-ignored — run train to recreate)
│   ├── plots/                       # All PNG figures
│   ├── predictions/
│   │   ├── best_model_predictions.csv   # Tuned LR predictions on test set
│   │   └── all_models_predictions.csv  # All 6 models side-by-side
│   └── report.html                  # Single-file report with all plots embedded
├── models/                          # Saved .joblib artifacts (git-ignored)
├── configs/config.yaml              # All hyperparameters, paths, Kaggle config
├── tests/test_preprocessing.py      # Pytest unit tests
├── main.py                          # CLI entry point
├── environment.yml                  # Conda environment (recommended)
└── requirements.txt                 # pip alternative
```

---

## Quick start

### 1. Set up the environment

**Option A — conda (recommended):**

```bash
conda env create -f environment.yml
conda activate shipment-disruption
```

**Option B — pip + virtualenv:**

```bash
pip install -r requirements.txt
```

### 2. Get the dataset

The dataset is included in this repository at `data/raw/`. If you want to pull it directly from Kaggle instead:

#### Set up Kaggle API credentials (one-time)

1. Go to [kaggle.com](https://www.kaggle.com) → Account → **API** → **Create New Token**
2. This downloads a `kaggle.json` file. Place it at:
   - **Windows:** `C:\Users\<your-username>\.kaggle\kaggle.json`
   - **Mac / Linux:** `~/.kaggle/kaggle.json`
3. Alternatively, export the credentials as environment variables:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

#### Find the correct dataset slug

The notebook this project is based on is at:  
`kaggle.com/code/nudratabbas/modeling-on-global-supply-chain-risk-logistics`

Open that page → click **Data** in the right sidebar → copy the dataset URL. It will look like:  
`kaggle.com/datasets/<owner>/<dataset-name>`

The slug is `<owner>/<dataset-name>`. Update `configs/config.yaml`:

```yaml
kaggle:
  dataset_slug: owner/dataset-name   # ← paste the slug here
  filename: global_supply_chain_risk_2026.csv
```

Then run:

```bash
python main.py download
```

To force a re-download even if the file already exists:

```bash
python main.py download --force
```

---

### 3. Run the pipeline

```bash
python main.py train
```

This will:
1. Check for the dataset (download from Kaggle if missing)
2. Engineer features, apply OHE, SMOTE, and StandardScaler
3. Train 6 classifiers with 5-fold cross-validation
4. Tune Logistic Regression with GridSearchCV (optimising recall)
5. Save `models/best_model.joblib` and `models/scaler.joblib`
6. Write to `output/`:
   - `plots/` — all EDA and evaluation figures as PNGs
   - `predictions/best_model_predictions.csv` — tuned LR predictions on the test set
   - `predictions/all_models_predictions.csv` — all 6 models' predictions side-by-side
   - `report.html` — **single self-contained file** with all plots, metrics tables, and predictions embedded inline

### 4. Score new shipments

```bash
python main.py predict --input path/to/new_shipments.csv
```

The input CSV must have the same columns as the training data. Output: predictions saved to `output/predictions/best_model_predictions.csv` and printed to the terminal.

### 5. Run tests

```bash
pytest tests/
```

### 6. EDA only

```bash
python main.py eda
```

---

## Dataset

**Source:** [Kaggle — Global Supply Chain Risk & Logistics](https://www.kaggle.com/code/nudratabbas/modeling-on-global-supply-chain-risk-logistics/input)  
5,000 shipment records × 14 features. No missing values.

| Feature | Type | Description |
|---|---|---|
| `Disruption_Occurred` | int (target) | 1 = disruption, 0 = no disruption |
| `Distance_km` | float | Route distance in kilometres |
| `Weight_MT` | float | Cargo weight (metric tonnes) |
| `Fuel_Price_Index` | float | Fuel cost index at departure |
| `Geopolitical_Risk_Score` | float | Route geopolitical risk (0–10) |
| `Carrier_Reliability_Score` | float | Historical carrier reliability (0–1) |
| `Lead_Time_Days` | float | Scheduled transit time |
| `Transport_Mode` | categorical | Air / Rail / Road / Sea |
| `Product_Category` | categorical | Electronics, Perishables, Pharmaceuticals, Automotive, Textiles |
| `Weather_Condition` | categorical | Clear / Fog / Rain / Storm / Hurricane |
| `Origin_Port` | categorical | 9 global ports |
| `Destination_Port` | categorical | 9 global ports |

**Engineered features:**

| Feature | Formula | Rationale |
|---|---|---|
| `Distance_per_Day` | `Distance_km / (Lead_Time_Days + 1)` | Transit speed — rushed routes correlate with disruptions |
| `Risk_Carrier_Interaction` | `Geopolitical_Risk × (1 − Carrier_Reliability)` | Compound risk on unreliable carriers through risky regions |
| `Heavy_Cargo` | 1 if `Weight_MT` > median | Binary flag for heavy shipments |
| `Month`, `Quarter` | Extracted from `Date` | Seasonal effects |

---

## Model results

| Model | Accuracy | Recall | F1 | ROC-AUC | CV AUC (5-fold) |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.741** | **0.770** | **0.785** | **0.818** | 0.878 |
| Decision Tree | 0.718 | 0.715 | 0.756 | 0.797 | 0.831 |
| Random Forest | 0.730 | 0.682 | 0.756 | 0.824 | 0.865 |
| Gradient Boosting | 0.721 | 0.742 | 0.765 | 0.804 | 0.867 |
| KNN | 0.658 | 0.666 | 0.705 | 0.721 | 0.814 |
| Naive Bayes | 0.604 | 0.362 | 0.529 | 0.799 | 0.844 |

Logistic Regression was selected as the final model and tuned with GridSearchCV, optimising recall — the cost of a missed disruption outweighs a false alarm in supply chain risk management.

**Top predictors** (by SHAP absolute impact):
1. `Weather_Condition_Hurricane` — 25.3%
2. `Weather_Condition_Storm` — 8.0%
3. `Geopolitical_Risk_Score` — 6.7%
4. `Distance_per_Day` (engineered) — 2.6%
5. `Carrier_Reliability_Score` — 2.4%

---

## Production roadmap

> This project was built on a static dataset. Turning it into a real-time operational tool is the natural next step. The recommendations below are drawn directly from our analysis and presentation.

### Live data integration

The current model was trained on historical shipment records. In production, it should score **shipments in real time** — ideally at the point of booking, before the shipment departs.

**Data sources to connect:**

| Source | What it provides | Integration path |
|---|---|---|
| TMS / ERP system | Shipment records (origin, destination, carrier, weight, lead time) | REST API or database read |
| Weather API (e.g. OpenWeatherMap, Tomorrow.io) | Live weather conditions along the route | HTTP polling at departure |
| Geopolitical risk feed (e.g. Control Risks, Verisk Maplecroft) | Real-time geopolitical risk scores by region | Scheduled sync |
| Carrier performance database | Up-to-date carrier reliability scores | Periodic refresh from TMS data |

**Suggested architecture:**

```
[TMS / ERP] ──► [Feature Pipeline] ──► [Trained Model] ──► [Risk Score]
                        ▲                                         │
               [Weather API]                              [Alert / Flag]
               [Risk Feed  ]                              [Dashboard   ]
```

- Replace the static CSV with a **live data connector** (`src/data/live_feed.py`)
- Score each shipment on creation using `main.py predict`
- Flag high-probability disruptions (e.g. `Disruption_Probability > 0.7`) as alerts in the TMS

### Dashboard

A real-time dashboard would make the model's output actionable for logistics managers and planners. Recommended metrics to surface:

- **Current high-risk shipments** — table of active shipments with disruption probability ≥ 0.7
- **Disruption rate trend** — rolling 30-day actual vs predicted disruption rate
- **Risk by route** — heatmap of origin → destination pairs by predicted risk score
- **Feature drivers** — top 3 risk factors per shipment (from SHAP values)
- **Model performance monitor** — actual vs predicted confusion matrix, updated as ground truth arrives

**Recommended stack:**
- [Streamlit](https://streamlit.io) — fastest path to a working dashboard, pure Python, easy to deploy
- [Grafana](https://grafana.com) — better for operational teams already using it, connects to SQL/InfluxDB
- [Tableau / Power BI](https://www.tableau.com) — preferred if stakeholders are in a business intelligence workflow

### Model retraining

A static model drifts over time as shipping patterns, geopolitics, and carrier networks change.

- **Schedule monthly retraining** using fresh TMS data via `python main.py train`
- Monitor **data drift** (compare incoming feature distributions to training distributions)
- Track **concept drift** — if real-world precision/recall degrades below threshold, trigger a retraining job
- Use a model registry (e.g. [MLflow](https://mlflow.org)) to version and compare models before promoting to production

### Business integration priorities (from our analysis)

1. **Carrier thresholds** — automatically flag carriers with `Carrier_Reliability_Score < 0.75` on high geopolitical-risk routes
2. **Weather rerouting** — trigger rerouting proposals when the model detects Hurricane or Storm conditions on active routes
3. **Perishables and Pharmaceuticals** — apply stricter carrier filters and enhanced monitoring for time-sensitive cargo categories
4. **Transit speed alerts** — flag shipments where `Distance_per_Day` exceeds historical norms for the route (a proxy for rushed, higher-risk schedules)

---

## Configuration

All tunable parameters live in `configs/config.yaml` — no need to edit source code.

```yaml
data:
  raw_path: data/raw/global_supply_chain_risk_2026.csv
  test_size: 0.2
  random_state: 42

kaggle:
  dataset_slug: owner/dataset-name   # update with the Kaggle slug
  filename: global_supply_chain_risk_2026.csv

models:
  logistic_regression:
    max_iter: 1000
  random_forest:
    n_estimators: 200
    max_depth: 10
  ...
```
