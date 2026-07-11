# 📈 End-to-End Sales Forecasting & Demand Intelligence System

A production-style data science pipeline built on the real **Kaggle Superstore Sales dataset**
(9,800 orders, Jan 2015 – Dec 2018) — covering time-series forecasting, anomaly detection,
product demand segmentation, and a live interactive dashboard.

🔗 **Live Dashboard:** https://salesforecasting-enwtcpseefq9uaryoz7wk3.streamlit.app/

---

## 📌 Project Overview

Retail and e-commerce companies live or die by one question: *how much of each product will
sell next month, and will there be enough stock to meet that demand?* This project builds an
intelligent forecasting system that:

- Predicts future product demand using 3 different forecasting approaches
- Detects unusual sales spikes/drops with two independent anomaly-detection methods
- Segments products into demand-behavior clusters for tailored stocking strategy
- Presents everything through a live, interactive Streamlit dashboard

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Data & Feature Engineering | Pandas, NumPy |
| Time-Series Analysis | Statsmodels (SARIMA, seasonal decomposition, ADF test) |
| Forecasting | SARIMA · Prophet · XGBoost |
| Anomaly Detection | Isolation Forest (scikit-learn), Z-score |
| Segmentation | K-Means, PCA (scikit-learn) |
| Visualization | Matplotlib, Seaborn |
| Dashboard / Deployment | Streamlit, Streamlit Community Cloud |

## 📂 Repository Structure

```
├── analysis.ipynb                     # Full 8-task analysis notebook (EDA → forecasting → clustering)
├── app.py                             # Streamlit dashboard (4 pages)
├── train.csv                          # Superstore Sales dataset (Kaggle)
├── vgsales.csv                        # Video Game Sales dataset (Task 1 multi-source merge exercise)
├── requirements.txt                   # Dependencies for the deployed Streamlit app
├── requirements-colab-notebook.txt    # Extra dependencies needed only for the notebook (Colab)
├── summary.docx                       # 2-page executive business report
└── charts/                            # Exported chart images (PNG)
```

## 📊 Dashboard Pages

1. **Sales Overview** — KPI cards, sales by year, monthly trend, region/category filters
2. **Forecast Explorer** — pick a category/region + forecast horizon, see the model's forecast, MAE/RMSE
3. **Anomaly Report** — flagged anomalous weeks with likely causes
4. **Demand Segments** — product sub-category clusters (K-Means + PCA)

## 🔍 Key Findings

- **Technology** is the top revenue category ($827K), narrowly ahead of Furniture and Office Supplies
- **East region** shows the most consistent year-over-year growth of any region
- **November, September, and December** are consistently the strongest months every year
- Of the 3 forecasting models compared on held-out real data, the **gradient-boosted tree model
  (XGBoost)** was the most accurate
- 13 anomalous sales weeks were detected; most align with the holiday season, but the single
  largest spike (March 2015) has no obvious seasonal driver and was flagged for manual review

Full findings, charts, and business recommendations are in `analysis.ipynb` and `summary.docx`.

## ▶️ Running Locally

```bash
git clone https://github.com/Asma957/Sales_Forecasting.git
cd Sales_Forecasting
pip install -r requirements.txt
streamlit run app.py
```

## 📓 Running the Notebook

`analysis.ipynb` runs end-to-end on `train.csv` / `vgsales.csv` already in this repo. For the
real `SARIMAX` / `Prophet` / `XGBRegressor` model cells in Task 3 (Google Colab recommended):

```bash
pip install -r requirements-colab-notebook.txt
```

## 📄 Dataset Sources

- Superstore Sales: https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting
- Video Game Sales: https://www.kaggle.com/datasets/gregorut/videogamesales

## 👤 Author

**Asma** — [GitHub Profile](https://github.com/Asma957)

---

*Built as part of a data science internship project — End-to-End Sales Forecasting & Demand
Intelligence System.*
