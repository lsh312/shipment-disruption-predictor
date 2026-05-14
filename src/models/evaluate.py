import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve
import shap

COLORS = ['#2e6da4', '#e67e22', '#c0392b', '#1a7a4a', '#8e44ad', '#2c3e50']


def print_results_table(results: dict) -> None:
    print(f"\n{'Model':<25} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'AUC':>6} {'CV AUC':>12}")
    print('-' * 75)
    for name, res in results.items():
        print(
            f"{name:<25} {res['accuracy']:>6.3f} {res['precision']:>6.3f} "
            f"{res['recall']:>7.3f} {res['f1']:>6.3f} {res['roc_auc']:>6.3f} "
            f"  {res['cv_mean']:.3f}±{res['cv_std']:.3f}"
        )


def plot_roc_curves(results: dict, y_test: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, res), col in zip(results.items(), COLORS):
        RocCurveDisplay.from_predictions(
            y_test, res['y_prob'],
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            ax=ax, color=col,
        )
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
    ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(results: dict, y_test: pd.Series) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=['No Disruption', 'Disruption']
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
    plt.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_metric_comparison(results: dict) -> plt.Figure:
    metrics = ['accuracy', 'f1', 'roc_auc', 'cv_mean']
    metric_labels = ['Accuracy', 'F1-Score', 'ROC-AUC', 'CV AUC (mean)']
    x = np.arange(len(metric_labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, col) in enumerate(zip(results.keys(), COLORS)):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=col, edgecolor='white', alpha=0.9)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    plt.tight_layout()
    return fig


def plot_shap(model, X_test_sc: np.ndarray, features: list) -> plt.Figure:
    X_df = pd.DataFrame(X_test_sc, columns=features)
    masker = shap.maskers.Independent(X_df)
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_vals = explainer.shap_values(X_df)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plt.sca(axes[0])
    shap.summary_plot(shap_vals, X_df, plot_type='bar', show=False, max_display=10)
    axes[0].set_title('SHAP — Mean Absolute Impact', fontweight='bold')
    plt.sca(axes[1])
    shap.summary_plot(shap_vals, X_df, show=False, max_display=10)
    axes[1].set_title('SHAP — Feature Impact Direction', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_learning_curve(model, X_train_res: pd.DataFrame, y_train_res: pd.Series,
                         cv, model_name: str, random_state: int = 42) -> plt.Figure:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_res, y_train_res,
        cv=cv, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True, n_jobs=-1, random_state=random_state,
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#2e6da4')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#c0392b')
    ax.plot(train_sizes, train_mean, 'o-', color='#2e6da4', label='Training Score')
    ax.plot(train_sizes, val_mean, 's-', color='#c0392b', label='Validation Score')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('ROC-AUC')
    ax.set_title(f'Learning Curve — {model_name}', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()

    gap = train_mean[-1] - val_mean[-1]
    print(f'Train-Validation gap at full data: {gap:.4f}')
    return fig
