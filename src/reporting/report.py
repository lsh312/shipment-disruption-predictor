"""
Generates a self-contained HTML report combining EDA plots, model evaluation,
SHAP analysis, and predictions — all charts are embedded as base64 so the file
is a single portable document.
"""
import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _img_tag(b64: str, alt: str = '') -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;margin:12px 0;">'


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; color: #2d2d2d; }
.container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
header { background: #1a3a5c; color: white; padding: 36px 24px; margin-bottom: 32px; }
header h1 { font-size: 1.8rem; margin-bottom: 6px; }
header p  { opacity: 0.8; font-size: 0.95rem; }
h2 { font-size: 1.3rem; color: #1a3a5c; border-left: 4px solid #2e6da4;
     padding-left: 12px; margin: 32px 0 16px; }
h3 { font-size: 1.05rem; color: #444; margin: 20px 0 10px; }
table { width: 100%; border-collapse: collapse; background: white;
        border-radius: 6px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
th { background: #2e6da4; color: white; padding: 10px 14px; text-align: left; font-size: .9rem; }
td { padding: 9px 14px; font-size: .88rem; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f9f9f9; }
.best td { font-weight: bold; background: #edf4ff !important; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-size: .78rem; font-weight: bold; }
.badge-blue { background: #dbeafe; color: #1e40af; }
.badge-green { background: #dcfce7; color: #15803d; }
.plot-grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
.card { background: white; border-radius: 8px; padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); }
footer { margin-top: 48px; padding: 16px 0; text-align: center;
         font-size: .82rem; color: #888; border-top: 1px solid #ddd; }
"""


def _metrics_table(results: dict, best_name: str) -> str:
    rows = ''
    for name, res in results.items():
        cls = ' class="best"' if name == best_name else ''
        badge = ' <span class="badge badge-green">best</span>' if name == best_name else ''
        rows += (
            f'<tr{cls}><td>{name}{badge}</td>'
            f'<td>{res["accuracy"]:.3f}</td><td>{res["precision"]:.3f}</td>'
            f'<td>{res["recall"]:.3f}</td><td>{res["f1"]:.3f}</td>'
            f'<td>{res["roc_auc"]:.3f}</td>'
            f'<td>{res["cv_mean"]:.3f} ± {res["cv_std"]:.3f}</td></tr>\n'
        )
    return f"""
<table>
  <thead><tr>
    <th>Model</th><th>Accuracy</th><th>Precision</th>
    <th>Recall</th><th>F1</th><th>ROC-AUC</th><th>CV AUC (5-fold)</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _predictions_table(predictions_df: pd.DataFrame, n: int = 25) -> str:
    subset = predictions_df.head(n)
    header = ''.join(f'<th>{c}</th>' for c in subset.columns)
    body = ''
    for _, row in subset.iterrows():
        cells = ''
        for val in row:
            if isinstance(val, float):
                cells += f'<td>{val:.4f}</td>'
            else:
                cells += f'<td>{val}</td>'
        body += f'<tr>{cells}</tr>\n'
    return f"""
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{body}</tbody>
</table>
<p style="font-size:.82rem;color:#888;margin-top:6px;">
  Showing first {min(n, len(predictions_df))} of {len(predictions_df)} rows.
</p>"""


def generate_report(
    results: dict,
    best_name: str,
    figures: dict,
    predictions_df: pd.DataFrame,
    output_path: str,
    dataset_info: dict = None,
) -> None:
    """
    figures: dict mapping section label → matplotlib Figure
    dataset_info: optional dict with keys 'n_rows', 'n_features', 'class_balance'
    """
    info = dataset_info or {}
    plots_html = ''
    for label, fig in figures.items():
        b64 = _fig_to_base64(fig)
        plots_html += f'<div class="card"><h3>{label}</h3>{_img_tag(b64, label)}</div>\n'

    dataset_section = ''
    if info:
        dataset_section = f"""
<h2>Dataset Overview</h2>
<div class="card">
  <p><strong>Records:</strong> {info.get('n_rows', '—')} &nbsp;|&nbsp;
     <strong>Features:</strong> {info.get('n_features', '—')} &nbsp;|&nbsp;
     <strong>Disruption rate:</strong> {info.get('class_balance', '—')}
  </p>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Supply Chain Disruption — ML Report</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Supply Chain Disruption Prediction</h1>
    <p>Machine Learning II &nbsp;·&nbsp; MBDS 2026 &nbsp;·&nbsp;
       Jan · Caspar · Lea · Ghezlan · Fouad</p>
    <p style="margin-top:8px;opacity:.65;font-size:.85rem;">
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
  </div>
</header>

<div class="container">

  {dataset_section}

  <h2>Model Performance Comparison</h2>
  {_metrics_table(results, best_name)}

  <h2>Evaluation Plots</h2>
  <div class="plot-grid">{plots_html}</div>

  <h2>Predictions on Test Set</h2>
  {_predictions_table(predictions_df)}

</div>
<footer>
  Supply Chain Disruption ML Pipeline &nbsp;·&nbsp; MBDS 2026
</footer>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report saved → {output_path}')
