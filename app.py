
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

DATA_DIR = "features"      # Preprocessed CSVs
MODEL_DIR = "predictions"        # Saved XGBoost models
COMPANIES = [f.replace("_features.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_features.csv")]
NEWS_API_KEY = "c09d9d4fb6394b76a3f64e17255348d9"

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Fincast")
st.subheader("Forecasting finance, empowering decisions.")
            
# Sidebar Inputs
company = st.sidebar.selectbox("Select Company", COMPANIES)
show_news = st.sidebar.checkbox("Show General Financial News")

# Load Data & Model
df = pd.read_csv(os.path.join(DATA_DIR, f"{company}_features.csv"), index_col="Date", parse_dates=True)
model = joblib.load(os.path.join(MODEL_DIR, f"{company}_xgb_model.pkl"))

# Feature Engineering 
df["Return"] = df["Close"].pct_change()
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_21"] = df["Close"].rolling(21).mean()
df["Volatility_7"] = df["Return"].rolling(7).std()
df["Volatility_21"] = df["Return"].rolling(21).std()
df["High_Low_Spread"] = df["High"] - df["Low"]
df["Open_Close_Change"] = df["Open"] - df["Close"]

# Lag features
for i in range(1, 31):
    df[f"Return_lag{i}"] = df["Return"].shift(i)

df.dropna(inplace=True)

# Exact feature order
feature_cols = ['High', 'Low', 'Open', 'Volume', 'MA_7', 'MA_21', 'Volatility_7',
                'Volatility_21', 'High_Low_Spread', 'Open_Close_Change'] + [f"Return_lag{i}" for i in range(1,31)]
X_latest = df[feature_cols].iloc[[-1]]

# Predict Next-Day Price
next_return = model.predict(X_latest)[0]
last_close = df["Close"].iloc[-1]
predicted_close = last_close * (1 + next_return)

st.subheader(f"{company} Next-Day Prediction")
col1, col2 = st.columns(2)
col1.metric(label="Predicted Close Price", value=f"₹ {predicted_close:.2f}")
col2.metric(label="Predicted Return", value=f"{next_return:.4%}")

# Show Today's OHLC
st.subheader(f"{company} Latest OHLC")
latest_ohlc = df[["Open", "High", "Low", "Close", "Volume"]].iloc[-1]
st.table(latest_ohlc.to_frame().T)

# Historical Price Graph 
st.subheader(f"{company} Historical Prices (Last 2 Years)")

last_two_years = df[df.index >= (df.index.max() - pd.DateOffset(years=2))]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last_two_years.index, 
    y=last_two_years["Close"], 
    mode="lines", 
    name="Close Price", 
    line=dict(color="blue")
))

fig.update_layout(
    title=f"{company} Close Price History (Last 2 Years)",
    xaxis_title="Date", 
    yaxis_title="Price",
    template="plotly_dark", 
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Feature Importance Plot
st.subheader("📊 Feature Importance (XGBoost)")
booster = model.get_booster()
importance = booster.get_score(importance_type="weight")
importance_df = pd.DataFrame({
    "Feature": list(importance.keys()),
    "Importance": list(importance.values())
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                 title="Feature Importance", color="Importance", height=600)
st.plotly_chart(fig_imp, use_container_width=True)

# General Financial News
if show_news:
    st.subheader("📰 General Financial News")
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=8&apiKey={NEWS_API_KEY}"
    #url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&language=en&pageSize=10&apiKey={NEWS_API_KEY}"

    response = requests.get(url)

    if response.status_code == 200:
        news_data = response.json().get("articles", [])
        if news_data:
            for article in news_data:
                with st.container():
                    st.markdown(f"### [{article['title']}]({article['url']})")
                    if article.get("urlToImage"):
                        st.image(article["urlToImage"], use_container_width=True)
                    st.write(article.get("description", ""))
                    st.caption(f"Source: {article['source']['name']} | Published: {datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%d %b %Y, %I:%M %p')}")
                    st.markdown("---")
        else:
            st.warning("No financial news available at the moment.")
    else:
        st.error("❌ Failed to fetch news. Check your API key.")

# Footer

st.markdown("""
---
<div style='text-align: center;'>
    <p>🚀 Made with ❤️ by <b>Mantra Gupta</b></p>
    <p>© No Copyright</p>
    <p>📊 Data Sources: Yahoo Finance, NewsAPI</p>
</div>
""", unsafe_allow_html=True)

# redeploy trigger Tue Aug 26 13:11:14 UTC 2025
# redeploy trigger Tue Aug 26 13:20:32 UTC 2025
# redeploy trigger Wed Aug 27 13:08:07 UTC 2025
# redeploy trigger Thu Aug 28 13:08:36 UTC 2025
# redeploy trigger Fri Aug 29 13:06:56 UTC 2025
# redeploy trigger Sat Aug 30 13:02:34 UTC 2025
# redeploy trigger Sun Aug 31 13:03:24 UTC 2025
# redeploy trigger Mon Sep  1 13:09:36 UTC 2025
# redeploy trigger Tue Sep  2 13:09:45 UTC 2025
# redeploy trigger Wed Sep  3 13:07:25 UTC 2025
# redeploy trigger Thu Sep  4 13:05:21 UTC 2025
# redeploy trigger Fri Sep  5 13:06:07 UTC 2025
# redeploy trigger Sat Sep  6 13:00:55 UTC 2025
# redeploy trigger Sun Sep  7 13:01:27 UTC 2025
# redeploy trigger Mon Sep  8 13:09:51 UTC 2025
# redeploy trigger Tue Sep  9 13:11:05 UTC 2025
# redeploy trigger Wed Sep 10 13:06:53 UTC 2025
# redeploy trigger Thu Sep 11 13:05:45 UTC 2025
# redeploy trigger Fri Sep 12 13:04:55 UTC 2025
# redeploy trigger Sat Sep 13 13:00:31 UTC 2025
# redeploy trigger Sun Sep 14 13:07:23 UTC 2025
# redeploy trigger Mon Sep 15 13:08:05 UTC 2025
# redeploy trigger Tue Sep 16 13:08:11 UTC 2025
# redeploy trigger Wed Sep 17 13:08:03 UTC 2025
# redeploy trigger Thu Sep 18 13:07:46 UTC 2025
# redeploy trigger Fri Sep 19 13:06:55 UTC 2025
# redeploy trigger Sat Sep 20 13:02:37 UTC 2025
