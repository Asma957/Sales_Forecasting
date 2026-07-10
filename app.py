"""
Sales Forecasting & Demand Intelligence Dashboard
===================================================
Run locally:      streamlit run app.py
Deploy free:       https://share.streamlit.io  (Streamlit Community Cloud)
                    -> connect your GitHub repo containing this file + requirements.txt + train.csv

Expects the following files in the same folder (produced by analysis.ipynb):
    train.csv                       (raw Superstore data)
    data/model_comparison.csv       (Task 3 output)
    data/anomalies_isoforest.csv    (Task 5 output)
    data/clusters.csv               (Task 6 output)
If these processed files are missing, the app falls back to computing them
live from train.csv so it still runs standalone.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide", page_icon="📈")
sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent
CSV_PATH = DATA_DIR / "train.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # Real Kaggle Superstore data uses DD/MM/YYYY; fall back gracefully if a different
    # export uses MM/DD/YYYY so this app doesn't break on either.
    try:
        df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    except ValueError:
        df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
        df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["ShipDays"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


def holt_winters_forecast(series, season_len=12, h=3, alpha=0.25, beta=0.05, gamma=0.35):
    y = series.values.astype(float)
    n = len(y)
    if n < season_len * 2:
        season_len = max(2, n // 2)
    season = np.array([y[i::season_len].mean() for i in range(season_len)])
    level = y[:season_len].mean()
    trend = (y[season_len:2*season_len].mean() - y[:season_len].mean()) / season_len if n >= 2 * season_len else 0.0
    seasonals = list(season)
    levels, trends = [level], [trend]
    for i in range(n):
        s_idx = i % season_len
        seas = seasonals[s_idx]
        prev_level, prev_trend = levels[-1], trends[-1]
        new_level = alpha * (y[i] - seas) + (1 - alpha) * (prev_level + prev_trend)
        new_trend = beta * (new_level - prev_level) + (1 - beta) * prev_trend
        new_seas = gamma * (y[i] - new_level) + (1 - gamma) * seas
        seasonals[s_idx] = new_seas
        levels.append(new_level); trends.append(new_trend)
    fc = [levels[-1] + k * trends[-1] + seasonals[(n + k - 1) % season_len] for k in range(1, h + 1)]
    return np.array(fc)


df = load_data()

st.sidebar.title("📈 Sales Intelligence")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} orders | {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")

# ============================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    c2.metric("Total Orders", f"{len(df):,}")
    c3.metric("Avg Order Value", f"${df['Sales'].mean():,.2f}")
    c4.metric("Avg Ship Days", f"{df['ShipDays'].mean():.1f}")

    st.markdown("### Filters")
    fc1, fc2 = st.columns(2)
    regions = fc1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    categories = fc2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    fdf = df[df["Region"].isin(regions) & df["Category"].isin(categories)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        yearly = fdf.groupby("Year")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=yearly.index, y=yearly.values, ax=ax, color="#2980b9")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = fdf.set_index("Order Date").resample("MS")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(monthly.index, monthly.values, color="#27ae60")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Region")
        reg = fdf.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=reg.values, y=reg.index, ax=ax, color="#8e44ad")
        st.pyplot(fig)
    with col4:
        st.subheader("Sales by Category")
        cat = fdf.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=cat.values, y=cat.index, ax=ax, color="#e67e22")
        st.pyplot(fig)

# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Best-performing model from Task 3 model comparison (lowest RMSE), applied to your chosen segment.")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    value = st.selectbox(f"Select {dim}", sorted(df[dim].unique()))
    horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    sub = df[df[dim] == value]
    monthly = sub.set_index("Order Date").resample("MS")["Sales"].sum()

    if len(monthly) >= 6:
        test_h = min(3, len(monthly) // 4)
        train, test = monthly.iloc[:-test_h], monthly.iloc[-test_h:]
        fc_test = holt_winters_forecast(train, h=test_h)
        mae = np.mean(np.abs(test.values - fc_test))
        rmse = np.sqrt(np.mean((test.values - fc_test) ** 2))

        fc_future = holt_winters_forecast(monthly, h=horizon)
        future_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly.index, monthly.values, label="Actual", color="#2c3e50")
        ax.plot(future_idx, fc_future, "--o", label="Forecast", color="#c0392b")
        ax.axvline(monthly.index[-1], color="gray", linestyle=":")
        ax.legend(); ax.set_ylabel("Sales ($)")
        ax.set_title(f"{value} — {horizon}-Month Forecast")
        st.pyplot(fig)

        fcol1, fcol2 = st.columns(2)
        fcol1.metric("Model MAE (backtest)", f"${mae:,.0f}")
        fcol2.metric("Model RMSE (backtest)", f"${rmse:,.0f}")

        st.subheader("Forecast values")
        st.dataframe(pd.DataFrame({"Date": future_idx.date, "Forecasted Sales": fc_future.round(2)}))
    else:
        st.warning("Not enough monthly history for this segment to forecast reliably.")

# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    from sklearn.ensemble import IsolationForest

    weekly = df.set_index("Order Date").resample("W")["Sales"].sum()
    X = np.hstack([weekly.values.reshape(-1, 1), weekly.rolling(4, min_periods=1).mean().values.reshape(-1, 1)])
    iso = IsolationForest(contamination=0.06, random_state=42)
    pred = iso.fit_predict(X)
    anomalies = weekly[pred == -1]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly.index, weekly.values, color="#34495e", label="Weekly Sales")
    ax.scatter(anomalies.index, anomalies.values, color="#e74c3c", s=70, zorder=5, label="Anomaly")
    ax.legend(); ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    st.dataframe(pd.DataFrame({"Week": anomalies.index.date, "Sales": anomalies.values.round(2)})
                 .sort_values("Sales", ascending=False))

# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "Demand Segments":
    st.title("Product Demand Segments")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    grp = df.groupby("Sub-Category")
    total_sales = grp["Sales"].sum()
    avg_order_value = grp["Sales"].mean()
    monthly_by_sub = df.set_index("Order Date").groupby("Sub-Category").resample("MS")["Sales"].sum()
    volatility = monthly_by_sub.groupby("Sub-Category").std()
    feat = pd.DataFrame({"total_sales": total_sales, "volatility": volatility, "avg_order_value": avg_order_value}).dropna()

    k = st.slider("Number of clusters (k)", 2, 6, 3)
    X = StandardScaler().fit_transform(feat)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    feat["cluster"] = km.labels_.astype(str)
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    feat["pc1"], feat["pc2"] = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(feat["cluster"].unique()):
        sub = feat[feat["cluster"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=140, label=f"Cluster {c}", edgecolor="black")
    for name, row in feat.iterrows():
        ax.annotate(name, (row["pc1"], row["pc2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-category → cluster mapping")
    st.dataframe(feat.reset_index().rename(columns={"index": "Sub-Category"})[["Sub-Category", "cluster", "total_sales", "volatility", "avg_order_value"]])
