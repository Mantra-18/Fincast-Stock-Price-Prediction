import pandas as pd
import os
import glob

# Input & Output folders
input_folder = "stock_data"
cleaned_folder = "cleaned"
features_folder = "features"

os.makedirs(cleaned_folder, exist_ok=True)
os.makedirs(features_folder, exist_ok=True)


# Loop through all CSV files in stock_data/
for file in glob.glob(os.path.join(input_folder, "*.csv")):
    try:
        company = os.path.splitext(os.path.basename(file))[0].upper()
        print(f"🔄 Processing {company}...")

        # Step 1: Load raw data & clean
        df = pd.read_csv(file, skiprows=3, header=None)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

        # Convert numeric columns
        numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # Save cleaned dataset
        cleaned_path = os.path.join(cleaned_folder, f"{company}_cleaned.csv")
        df.to_csv(cleaned_path)

        # Step 2: Feature Engineering
        df["Return"] = df["Close"].pct_change()
        df["MA_7"] = df["Close"].rolling(window=7).mean()
        df["MA_21"] = df["Close"].rolling(window=21).mean()
        df["Volatility_7"] = df["Close"].rolling(window=7).std()
        df["Volatility_21"] = df["Close"].rolling(window=21).std()
        df["High_Low_Spread"] = df["High"] - df["Low"]
        df["Open_Close_Change"] = df["Open"] - df["Close"]

        df.dropna(inplace=True)

        # Save feature-engineered dataset
        feature_path = os.path.join(features_folder, f"{company}_features.csv")
        df.to_csv(feature_path)

        print(f" Completed {company}: Saved {cleaned_path} and {feature_path}\n")

    except Exception as e:
        print(f" Error processing {file}: {e}")

print("🎉 Preprocessing & Feature Engineering Completed for all companies in stock_data/")
