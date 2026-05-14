import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


NUM_COLS = [
    'Distance_km', 'Weight_MT', 'Fuel_Price_Index',
    'Geopolitical_Risk_Score', 'Carrier_Reliability_Score', 'Lead_Time_Days',
]
CAT_COLS = ['Transport_Mode', 'Product_Category', 'Weather_Condition']
TARGET = 'Disruption_Occurred'
OVERALL_RATE = 61.3


def _setup():
    sns.set_theme(style='whitegrid', palette='muted')
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11


def plot_target_distribution(df: pd.DataFrame) -> plt.Figure:
    counts = df[TARGET].value_counts().sort_index()
    colors = ['#2e6da4', '#c0392b']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(['No Disruption (0)', 'Disruption (1)'], counts.values,
                color=colors, edgecolor='white', width=0.5)
    axes[0].set_title('Target Variable Distribution')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    axes[1].pie(counts.values, labels=['No Disruption', 'Disruption'],
                colors=colors, autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Class Balance')
    plt.suptitle('Disruption_Occurred — Target Variable', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_numerical_distributions(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUM_COLS):
        axes[i].hist(df[df[TARGET] == 0][col], bins=30, alpha=0.6,
                     label='No Disruption', color='#2e6da4')
        axes[i].hist(df[df[TARGET] == 1][col], bins=30, alpha=0.6,
                     label='Disruption', color='#c0392b')
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)
        axes[i].set_ylabel('Count')
    plt.suptitle('Numerical Feature Distributions by Target Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    corr = df[NUM_COLS + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, square=True)
    ax.set_title('Correlation Matrix — Numerical Features', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_categorical_disruption_rates(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, col in enumerate(CAT_COLS):
        ct = df.groupby(col)[TARGET].mean().sort_values(ascending=False)
        bars = axes[i].bar(ct.index, ct.values * 100,
                           color=['#c0392b' if v > 0.5 else '#2e6da4' for v in ct.values],
                           edgecolor='white')
        axes[i].axhline(y=OVERALL_RATE, color='gray', linestyle='--',
                        linewidth=1, label=f'Overall avg ({OVERALL_RATE}%)')
        axes[i].set_title(f'Disruption Rate by {col}')
        axes[i].set_ylabel('Disruption Rate (%)')
        axes[i].set_ylim(0, 100)
        axes[i].legend(fontsize=8)
        axes[i].tick_params(axis='x', rotation=30)
        for bar in bars:
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                         f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    plt.suptitle('Disruption Rate by Categorical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_port_analysis(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ['Origin_Port', 'Destination_Port']):
        ct = df.groupby(col)[TARGET].mean().sort_values(ascending=True)
        ax.barh(ct.index, ct.values * 100, color='#2e6da4', edgecolor='white')
        ax.axvline(x=OVERALL_RATE, color='#c0392b', linestyle='--',
                   linewidth=1.5, label=f'Avg {OVERALL_RATE}%')
        ax.set_title(f'Disruption Rate by {col}')
        ax.set_xlabel('Disruption Rate (%)')
        ax.set_xlim(0, 100)
        ax.legend()
    plt.suptitle('Disruption Rate by Port', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def run_eda(df: pd.DataFrame) -> None:
    _setup()
    plot_target_distribution(df)
    plot_numerical_distributions(df)
    plot_correlation_heatmap(df)
    plot_categorical_disruption_rates(df)
    plot_port_analysis(df)
    plt.show()
