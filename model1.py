import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

input_dir = "features"    
output_dir = "walk_forward_results"
os.makedirs(output_dir, exist_ok=True)

# Function to create lag features
def create_lags(data, col="Return", n_lags=30):
    for lag in range(1, n_lags + 1):
        data[f"{col}_lag{lag}"] = data[col].shift(lag)
    return data


feature_files = [f for f in os.listdir(input_dir) if f.endswith("_features.csv")]

for file in feature_files:
    company_name = file.replace("_features.csv", "")
    input_path = os.path.join(input_dir, file)
    print(f"\n📊 Processing {company_name}...")

    try:
        df = pd.read_csv(input_path, index_col="Date", parse_dates=True)

        # Target = Returns
        df["Return"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        # Lag Features
        df = create_lags(df, "Return", n_lags=30)
        df.dropna(inplace=True)

        # Features & Target
        X = df.drop(columns=["Close", "Return"])
        y = df["Return"]


        # Walk-Forward Validation
        window_size = int(len(X) * 0.7)  # start with 70% train
        test_size = 30                   # forecast horizon = 30 trading days
        step_size = 30                   # roll forward every 30 days

        rmse_scores, mae_scores, r2_scores = [], [], []
        preds, actuals, dates = [], [], []

        for start in range(0, len(X) - window_size - test_size, step_size):
            train_end = start + window_size
            test_end = train_end + test_size

            X_train, y_train = X.iloc[start:train_end], y.iloc[start:train_end]
            X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

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

            # Save metrics
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

            preds.extend(y_pred)
            actuals.extend(y_test.values)
            dates.extend(y_test.index)

     
        print("✅ Walk-Forward Validation Results:")
        print(f"Avg RMSE: {np.mean(rmse_scores):.6f}")
        print(f"Avg MAE:  {np.mean(mae_scores):.6f}")
        print(f"Avg R²:   {np.mean(r2_scores):.6f}")

        # Save results to CSV
        results_df = pd.DataFrame({
            "Date": dates,
            "Actual_Return": actuals,
            "Predicted_Return": preds
        })
        results_df.to_csv(os.path.join(output_dir, f"{company_name}_walk_forward_predictions.csv"), index=False)

      
        plt.figure(figsize=(12,6))
        plt.plot(dates, actuals, label="Actual Returns", color="blue", alpha=0.6)
        plt.plot(dates, preds, label="Predicted Returns", color="red", alpha=0.7)
        plt.title(f"{company_name} - Walk-Forward Predicted vs Actual Returns")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{company_name}_walk_forward_plot.png"))
        plt.close()

    except Exception as e:
        print(f"⚠️ Failed processing {company_name}: {e}")

print("\n Walk-forward validation completed for all companies!")
