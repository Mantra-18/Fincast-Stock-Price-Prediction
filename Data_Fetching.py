import yfinance as yf
import pandas as pd
import os

companies = {
    "Reliance": "RELIANCE.NS",
    "Bajaj_Finance": "BAJFINANCE.NS",
    "BPCL": "BPCL.NS",
    "Adani": "ADANIENT.NS",
    "Mahindra_Mahindra": "M&M.NS",
    "ONGC": "ONGC.NS",
    "Maruti_Suzuki": "MARUTI.NS",
    "Nestle_India": "NESTLEIND.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel":"BHARTIARTL.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Larsen & Toubro": "LT.NS"
    
}

# Directory to save CSVs
output_dir = "stock_data"
os.makedirs(output_dir, exist_ok=True)

# Fetch data and save CSV
for company_name, ticker in companies.items():
    print(f"Fetching data for {company_name} ({ticker})...")
    
    df = yf.download(ticker, start="2020-01-01", end="2025-08-23")
    
    if not df.empty:
        # Reset index to have 'Date' as a column
        df.reset_index(inplace=True)
        file_path = os.path.join(output_dir, f"{company_name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
    else:
        print(f"⚠️ No data fetched for {company_name}")

print("All done!")
