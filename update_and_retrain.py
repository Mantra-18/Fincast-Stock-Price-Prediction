import os
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBRegressor
from datetime import datetime, timedelta

DATA_DIR = "features"
MODEL_DIR = "predictions"
RAW_DIR = "stock_data"

COMPANY_TICKERS = {
    "Reliance": "RELIANCE.NS",
    "Adani": "ADANIENT.NS",
    "Mahindra_Mahindra": "M&M.NS",
    "Maruti_Suzuki": "MARUTI.NS",
    "Nestle_India": "NESTLEIND.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS"
    
}


def update_stock_data(company, ticker):
    """Fetch latest stock data and update CSV"""
    raw_file = os.path.join(RAW_DIR, f"{company}.csv")

    # load old data if exists
    if os.path.exists(raw_file):
        df_old = pd.read_csv(raw_file, parse_dates=["Date"], index_col="Date")
        last_date = df_old.index.max()
    else:
        df_old = pd.DataFrame()
        last_date = datetime.now() - timedelta(days=365*5)  # 5 years back

    # fetch new data from Yahoo Finance
    df_new = yf.download(ticker, start=last_date + timedelta(days=1))
    if not df_new.empty:
        df_new.reset_index(inplace=True)
        df_new.to_csv(raw_file, index=False, mode="a", header=not os.path.exists(raw_file))

    print(f"Updated stock data for {company}")
    return pd.read_csv(raw_file, parse_dates=["Date"], index_col="Date")


def create_features(df):
    """Generate features for training"""

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop bad rows 
    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Feature engineering
    df["Return"] = df["Close"].pct_change()
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_21"] = df["Close"].rolling(21).mean()
    df["Volatility_7"] = df["Return"].rolling(7).std()
    df["Volatility_21"] = df["Return"].rolling(21).std()
    df["High_Low_Spread"] = df["High"] - df["Low"]
    df["Open_Close_Change"] = df["Open"] - df["Close"]

    for i in range(1, 31):
        df[f"Return_lag{i}"] = df["Return"].shift(i)

    df.dropna(inplace=True)
    return df



def train_model(company, df):
    """Train XGB model for company and save"""
    feature_cols = ['High', 'Low', 'Open', 'Volume', 'MA_7', 'MA_21',
                    'Volatility_7','Volatility_21','High_Low_Spread',
                    'Open_Close_Change'] + [f"Return_lag{i}" for i in range(1,31)]

    X = df[feature_cols]
    y = df["Return"]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(MODEL_DIR, f"{company}_xgb_model.pkl")
    joblib.dump(model, model_path)

    print(f" Trained model saved for {company}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    for company, ticker in COMPANY_TICKERS.items():
        df = update_stock_data(company, ticker)     # Step 1: Update stock data
        df_feat = create_features(df)               # Step 2: Create features
        feat_file = os.path.join(DATA_DIR, f"{company}_features.csv")
        df_feat.to_csv(feat_file)                   # Save features
        train_model(company, df_feat)               # Step 3: Retrain & save model

    print("🎉 All companies updated and retrained!")


if __name__ == "__main__":
    main()
