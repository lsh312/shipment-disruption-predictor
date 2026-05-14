import argparse
import warnings
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from src.data.preprocessing import (
    load_data, engineer_features, encode_features,
    build_feature_list, prepare_data,
)
from src.models.train import (
    get_model_configs, train_and_evaluate, tune_logistic_regression,
)
from src.models.evaluate import (
    print_results_table, plot_roc_curves, plot_confusion_matrices,
    plot_metric_comparison, plot_shap, plot_learning_curve,
)
from src.models.predict import load_model, load_scaler, predict
from src.visualization.plots import (
    plot_target_distribution, plot_numerical_distributions,
    plot_correlation_heatmap, plot_categorical_disruption_rates,
    plot_port_analysis,
)
from src.reporting.report import generate_report
from src.data.ingestion import ensure_data_exists

warnings.filterwarnings('ignore')


def load_config(path: str = 'configs/config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _build_predictions_df(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: dict,
    base_features: list,
    best_name: str,
) -> tuple:
    """Returns (all_models_df, best_model_df) with predictions per model."""
    readable = [c for c in X_test.columns if c in base_features]
    base_df = X_test[readable].copy()
    base_df['Actual_Disruption'] = y_test.values

    all_df = base_df.copy()
    for name, res in results.items():
        col_prefix = name.replace(' ', '_')
        all_df[f'{col_prefix}_pred'] = res['y_pred']
        all_df[f'{col_prefix}_prob'] = res['y_prob'].round(4)

    best_df = base_df.copy()
    bp = results[best_name]
    best_df['Predicted_Disruption'] = bp['y_pred']
    best_df['Disruption_Probability'] = bp['y_prob'].round(4)

    return all_df, best_df


def cmd_download(config: dict, force: bool = False) -> None:
    ensure_data_exists(config, force=force)


def cmd_train(config: dict) -> None:
    plots_dir = config['output']['plots_dir']
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    Path(config['output']['dir']).mkdir(parents=True, exist_ok=True)
    Path(config['artifacts']['models_dir']).mkdir(exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print('Loading and preprocessing data...')
    ensure_data_exists(config)
    df_raw = load_data(config['data']['raw_path'])
    df = engineer_features(df_raw)
    df = encode_features(df, config['features']['encode_cols'])
    features = build_feature_list(
        df, config['features']['base'], config['features']['encode_cols']
    )
    n_ohe = len(features) - len(config['features']['base'])
    print(
        f'  Features: {len(features)} '
        f'({len(config["features"]["base"])} numeric + {n_ohe} OHE dummies)'
    )

    (X_train_res, X_test, y_train_res, y_test,
     X_train_sc, X_test_sc, scaler) = prepare_data(
        df, features, config['features']['target'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
    )
    print(f'  Train: {len(X_train_res)} (after SMOTE) | Test: {len(X_test)}')

    cv = StratifiedKFold(
        n_splits=config['cv']['n_splits'],
        shuffle=True,
        random_state=config['data']['random_state'],
    )

    # ── 2. EDA plots ──────────────────────────────────────────────────────────
    print('\nGenerating EDA plots...')
    eda_figs = {
        'Target Distribution':
            plot_target_distribution(df_raw),
        'Numerical Feature Distributions':
            plot_numerical_distributions(df_raw),
        'Correlation Heatmap':
            plot_correlation_heatmap(df_raw),
        'Disruption Rate by Category':
            plot_categorical_disruption_rates(df_raw),
        'Disruption Rate by Port':
            plot_port_analysis(df_raw),
    }
    for label, fig in eda_figs.items():
        _save_fig(fig, f'{plots_dir}/{label.lower().replace(" ", "_")}.png')

    # ── 3. Train base models ──────────────────────────────────────────────────
    print('\nTraining base models...')
    model_configs = get_model_configs(
        random_state=config['data']['random_state']
    )
    results = train_and_evaluate(
        model_configs, X_train_res, X_test,
        y_train_res, y_test, X_train_sc, X_test_sc, cv,
    )
    print_results_table(results)

    # ── 4. Tune Logistic Regression ───────────────────────────────────────────
    best_name = 'Logistic Regression'
    print(f'\nTuning {best_name}...')
    best_model, best_params, best_cv = tune_logistic_regression(
        X_train_sc, y_train_res, cv,
        random_state=config['data']['random_state'],
        scoring=config['tuning']['logistic_regression']['scoring'],
    )
    print(f'  Best CV Recall: {best_cv:.4f} | Params: {best_params}')

    y_pred_t = best_model.predict(X_test_sc)
    y_prob_t = best_model.predict_proba(X_test_sc)[:, 1]
    results[best_name].update({
        'model':     best_model,
        'y_pred':    y_pred_t,
        'y_prob':    y_prob_t,
        'accuracy':  round(accuracy_score(y_test, y_pred_t), 4),
        'precision': round(precision_score(y_test, y_pred_t), 4),
        'recall':    round(recall_score(y_test, y_pred_t), 4),
        'f1':        round(f1_score(y_test, y_pred_t), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob_t), 4),
    })

    # ── 5. Save model artifacts ───────────────────────────────────────────────
    joblib.dump(best_model, config['artifacts']['best_model_path'])
    joblib.dump(scaler, config['artifacts']['scaler_path'])
    print(f'\nSaved model  → {config["artifacts"]["best_model_path"]}')
    print(f'Saved scaler → {config["artifacts"]["scaler_path"]}')

    # ── 6. Evaluation plots ───────────────────────────────────────────────────
    print('\nGenerating evaluation plots...')
    eval_figs = {
        'ROC Curves':
            plot_roc_curves(results, y_test),
        'Confusion Matrices':
            plot_confusion_matrices(results, y_test),
        'Metric Comparison':
            plot_metric_comparison(results),
        'SHAP Feature Importance':
            plot_shap(best_model, X_test_sc, features),
        'Learning Curve':
            plot_learning_curve(
                best_model, X_train_res, y_train_res, cv,
                best_name, config['data']['random_state'],
            ),
    }
    for label, fig in eval_figs.items():
        _save_fig(fig, f'{plots_dir}/{label.lower().replace(" ", "_")}.png')

    # ── 7. Save predictions CSVs ──────────────────────────────────────────────
    pred_dir = Path(config['output']['predictions_path']).parent
    pred_dir.mkdir(parents=True, exist_ok=True)

    all_preds_df, best_preds_df = _build_predictions_df(
        X_test, y_test, results, config['features']['base'], best_name
    )

    all_path = str(pred_dir / 'all_models_predictions.csv')
    best_path = config['output']['predictions_path']
    all_preds_df.to_csv(all_path, index=False)
    best_preds_df.to_csv(best_path, index=False)
    print(f'Saved all-models predictions → {all_path}')
    print(f'Saved best-model predictions → {best_path}')

    # ── 8. Generate HTML report ───────────────────────────────────────────────
    all_figs = {**eda_figs, **eval_figs}
    dataset_info = {
        'n_rows': len(df_raw),
        'n_features': len(features),
        'class_balance': (
            f'{df_raw["Disruption_Occurred"].mean() * 100:.1f}% disruption'
        ),
    }
    generate_report(
        results=results,
        best_name=best_name,
        figures=all_figs,
        predictions_df=all_preds_df,
        output_path=config['output']['report_path'],
        dataset_info=dataset_info,
    )


def cmd_predict(config: dict, input_path: str) -> None:
    model = load_model(config['artifacts']['best_model_path'])
    scaler = load_scaler(config['artifacts']['scaler_path'])

    df = load_data(input_path)
    df = engineer_features(df)
    df = encode_features(df, config['features']['encode_cols'])
    features = build_feature_list(
        df, config['features']['base'], config['features']['encode_cols']
    )

    output = predict(model, df[features], scaler=scaler)

    pred_path = config['output']['predictions_path']
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)

    out_df = df[config['features']['base']].copy()
    if 'Shipment_ID' in df.columns:
        out_df.insert(0, 'Shipment_ID', df['Shipment_ID'].values)
    out_df['Predicted_Disruption'] = output['predictions']
    out_df['Disruption_Probability'] = output['probabilities'].round(4)
    out_df.to_csv(pred_path, index=False)
    print(out_df.to_string(index=False))
    print(f'\nSaved predictions → {pred_path}')


def cmd_eda(config: dict) -> None:
    from src.visualization.plots import run_eda
    df = load_data(config['data']['raw_path'])
    run_eda(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Supply Chain Disruption ML Pipeline'
    )
    parser.add_argument(
        'command', choices=['download', 'train', 'predict', 'eda'],
        help=(
            'download: fetch dataset from Kaggle | '
            'train: full pipeline + HTML report | '
            'predict: score new data | '
            'eda: EDA plots only'
        ),
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if the dataset already exists locally',
    )
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--input', help='CSV path required for predict')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == 'download':
        cmd_download(config, force=args.force)
    elif args.command == 'train':
        cmd_train(config)
    elif args.command == 'predict':
        if not args.input:
            parser.error('--input <path> is required for the predict command')
        cmd_predict(config, args.input)
    elif args.command == 'eda':
        cmd_eda(config)


if __name__ == '__main__':
    main()
