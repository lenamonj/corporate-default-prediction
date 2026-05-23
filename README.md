# Corporate Default Prediction

XGBoost classifier predicting US public company bankruptcy from accounting fundamentals. Trained on 78,682 firm-year observations from NYSE/NASDAQ companies (1999-2018) with temporal train/test splits and SHAP interpretability.

## What It Does

Takes raw accounting data (assets, liabilities, income, cash flow) for US public companies, engineers standard credit ratios (leverage, profitability, coverage), and trains an XGBoost classifier to predict bankruptcy. The model uses temporal validation - training on 1999-2011, validating on 2012-2014, and testing on 2015-2018 - so it must generalize across credit cycles. Class imbalance is handled via `scale_pos_weight`. Feature importance is explained with SHAP.

## Dataset

[American Companies Bankruptcy Prediction](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) - 78,682 firm-year observations from 8,262 NYSE/NASDAQ companies. CC0 Public Domain license.

## How to Run

1. Clone and install dependencies:
```bash
git clone https://github.com/lenamonj/corporate-default-prediction.git
cd corporate-default-prediction
pip install -r requirements.txt
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset) and unzip into `data/`.

3. Run the notebook:
```bash
jupyter notebook corporate_default_prediction.ipynb
```

## Engineered Features

| Ratio | Formula | Measures |
|-------|---------|----------|
| current_ratio | Current Assets / Total Assets | Liquidity |
| debt_to_assets | Long-Term Debt / Total Assets | Leverage |
| net_margin | Net Income / Net Sales | Profitability |
| ebitda_margin | EBITDA / Net Sales | Operating efficiency |
| roa | Net Income / Total Assets | Return on assets |
| gross_margin | Gross Profit / Net Sales | Pricing power |
| asset_turnover | Net Sales / Total Assets | Asset utilization |
| debt_to_ebitda | Long-Term Debt / EBITDA | Debt serviceability |

## License

MIT
