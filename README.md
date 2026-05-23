<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-Classifier-EC6C37?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Data-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/SHAP-Explainability-4B0082?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">Corporate Default Prediction</h1>

<p align="center">
  <strong>An XGBoost classifier that predicts US public company bankruptcy from accounting fundamentals - trained on 78,682 firm-year observations from NYSE/NASDAQ companies (1999-2018) with temporal validation and SHAP interpretability.</strong>
</p>

<p align="center">
  <em>No Bloomberg terminal. No paid data feeds. Just a free Kaggle dataset and scikit-learn.</em>
</p>

---

## Why This Exists

Predicting corporate default is one of the core problems in credit analysis. Most models are locked behind expensive platforms or proprietary datasets. This project demonstrates the full workflow using entirely free, public data - from raw accounting items to a production-quality classifier with interpretable feature importance.

The model uses a **temporal train/test split** (train on 1999-2011, test on 2015-2018), which is the only honest way to validate a financial model. No random shuffling, no data leakage, no inflated metrics.

---

## What It Produces

A single Jupyter notebook that runs the complete pipeline:

| Stage | What Happens |
|-------|-------------|
| **Data Loading** | 78,682 firm-year observations, 18 accounting variables, binary default label |
| **EDA** | Class balance, feature distributions by default status, correlation matrix |
| **Feature Engineering** | 8 credit ratios derived from raw accounting data (see table below) |
| **Temporal Split** | Train 1999-2011, Validate 2012-2014, Test 2015-2018 |
| **XGBoost** | Gradient-boosted classifier with `scale_pos_weight` for class imbalance |
| **Evaluation** | ROC-AUC, Precision-Recall curve, confusion matrix, classification report |
| **Interpretability** | SHAP summary and bar plots showing top 15 feature contributions |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/lenamonj/corporate-default-prediction.git
cd corporate-default-prediction
pip install -r requirements.txt
```

### 2. Get the data

Download from [Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) and unzip into `data/`.

### 3. Run

```bash
jupyter notebook corporate_default_prediction.ipynb
```

---

## Engineered Features

Standard credit ratios computed from raw accounting data:

| Ratio | Formula | Measures |
|-------|---------|----------|
| `current_ratio` | Current Assets / Total Assets | Liquidity |
| `debt_to_assets` | Long-Term Debt / Total Assets | Leverage |
| `net_margin` | Net Income / Net Sales | Profitability |
| `ebitda_margin` | EBITDA / Net Sales | Operating efficiency |
| `roa` | Net Income / Total Assets | Return on assets |
| `gross_margin` | Gross Profit / Net Sales | Pricing power |
| `asset_turnover` | Net Sales / Total Assets | Asset utilization |
| `debt_to_ebitda` | Long-Term Debt / EBITDA | Debt serviceability |

---

## Design Decisions

- **Temporal validation** - Train/test split by year, not random. A credit model that can't generalize across time is useless.
- **scale_pos_weight over SMOTE** - Cleaner than synthetic oversampling, avoids introducing artificial data points into the training set.
- **SHAP over feature importance** - XGBoost's built-in feature importance is biased toward high-cardinality features. SHAP values are theoretically grounded and show directional effects.
- **Outlier clipping at 1st/99th percentile** - Financial data has extreme tails. Winsorizing prevents a handful of outliers from dominating the model.
- **Credit ratios as features** - Raw accounting items are scale-dependent. Ratios normalize across company size and are the standard language of credit analysis.

---

## Dataset

[American Companies Bankruptcy Prediction](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)

- 78,682 firm-year observations from 8,262 NYSE/NASDAQ companies
- 18 accounting variables per observation
- Binary target: `alive` / `failed` (6.6% default rate)
- Period: 1999-2018
- License: CC0 Public Domain

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `xgboost` | Gradient-boosted classifier |
| `scikit-learn` | Metrics, train/test utilities |
| `shap` | Model interpretability |
| `matplotlib` | Plotting |
| `seaborn` | Heatmaps and styled plots |

---

## Project Structure

```
.
├── corporate_default_prediction.ipynb    # Full pipeline - one notebook
├── data/                                 # Kaggle dataset (not tracked in git)
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # You are here
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with Python, XGBoost, and public data. No proprietary tools required.</sub>
</p>
