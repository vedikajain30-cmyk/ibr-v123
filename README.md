# Valuation Impact of Capital Expansion Cycle on Indian Infrastructure Sector
**MS25GF013 | MGB October 2025 | SP Jain School of Global Management**

## Live Dashboard
Deployed on Streamlit Community Cloud.

## What This Dashboard Contains

| Page | Content |
|------|---------|
| 🏠 Overview & Macro Cycle | Year-wise cycle analysis, Gov Capex vs Mkt Cap, Correlation heatmap |
| 📊 Sub-Sector Efficiency (H4) | Efficiency ratios, OLS regression slopes, scatter plots |
| 🏢 Company Deep Dive | 10-year financial profile for all 25 companies |
| 📐 Leverage Tipping Point (H5) | Quadratic regression, D/E tipping point at 3.17x |
| 🤖 ML Classification & Rules | Random Forest / GB / LR · Accuracy · Precision · Recall · F1 · ROC-AUC · Feature Importance · Confidence & Lift |

## Dataset
- **25 Companies** | **10 Fiscal Years** (FY2016–FY2025) | **250 Observations**
- Sub-Sectors: Railways, Power, Roads, EPC, Ports, Airports, Utilities, Water

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1. Fork this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → `app.py` → Deploy
