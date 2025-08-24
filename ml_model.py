import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

input_dir = "features"   
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

# List all feature CSVs
feature_files = [f for f in os.listdir(input_dir) if f.endswith("_features.csv")]

# Function to create lag features
def create_lags(data, col="Return", n_lags=30):
    for lag in range(1, n_lags + 1):
        data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data

# Loop through all companies
for file in feature_files:
    company_name = file.replace("_features.csv", "")
    input_path = os.path.join(input_dir, file)
    print(f"\n🚀 Processing {company_name}...")

    try:
        # Load feature engineered dataset
        df = pd.read_csv(input_path, index_col="Date", parse_dates=True)

        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        # Create lag features of returns
        df = create_lags(df, "Return", n_lags=30)
        df.dropna(inplace=True)

        # Features & Target
        X = df.drop(columns=["Close", "Return"])
        y = df["Return"]

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=5)
        rmse_scores, mae_scores, r2_scores = [], [], []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        print(" Cross-Validation Results (5 folds):")
        print(f"   Avg RMSE: {np.mean(rmse_scores):.6f}")
        print(f"   Avg MAE:  {np.mean(mae_scores):.6f}")
        print(f"   Avg R²:   {np.mean(r2_scores):.6f}")

        # Train final model on full data
        final_model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        final_model.fit(X, y)

        # Backtest: Predict Returns → Prices
        pred_returns = final_model.predict(X)
        pred_prices = [df["Close"].iloc[0]]
        for r in pred_returns[1:]:
            pred_prices.append(pred_prices[-1] * (1 + r))

        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Close"], label="Actual Price", color="blue")
        plt.plot(df.index, pred_prices, label="Predicted Price", color="red", alpha=0.7)
        plt.title(f"{company_name} Stock Price Prediction (XGBoost on Return Lags)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{company_name}_price_plot.png"))
        plt.close()

        # Feature Importance
        importance = final_model.feature_importances_
        features = X.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features)
        plt.title(f"{company_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{company_name}_feature_importance.png"))
        plt.close()

        # One-Day Ahead Forecast
        X_latest = X.iloc[[-1]]
        next_return = final_model.predict(X_latest)[0]
        last_close = df["Close"].iloc[-1]
        predicted_close = last_close * (1 + next_return)

        print(f"\nLast available date: {df.index[-1].date()}")
        print(f"Predicted next-day return: {round(next_return, 6)}")
        print(f"Predicted next-day close price: {round(predicted_close, 2)}")

        # Save prediction to CSV
        forecast_df = pd.DataFrame({
            "Last_Date": [df.index[-1]],
            "Predicted_Return": [next_return],
            "Predicted_Close": [predicted_close]
        })
        forecast_df.to_csv(os.path.join(output_dir, f"{company_name}_next_day_prediction.csv"), index=False)
        print(f"Saved prediction CSV for {company_name}")

        # Save trained model
        model_path = os.path.join(output_dir, f"{company_name}_xgb_model.pkl")
        joblib.dump(final_model, model_path)
        print(f"Saved trained model for {company_name} at {model_path}")

    except Exception as e:
        print(f"Failed processing {company_name}: {e}")

print("\nAll companies processed successfully!")
