#!/usr/bin/env python3
"""
Main script to run housing price regression model
Filters by Bay Area metros: SF-Oakland-Berkeley and San Jose-Sunnyvale-Santa Clara
"""

from time_series_preparation import TimeSeriesDataPreparation
from regression_model import HousingPriceRegressor
from pathlib import Path

def main():
    print("="*70)
    print("HOUSING PRICE REGRESSION - BAY AREA METROS")
    print("="*70)
    
    # Step 1: Prepare time series data
    print("\n[1/2] Preparing time series data...")
    print("-" * 70)
    prep = TimeSeriesDataPreparation(
        data_dir="raw_data",
        eda_dir="../eda"
    )
    
    # Prepare data (predicting 1 month ahead)
    processed_data = prep.save_processed_data(
        "data/processed_timeseries_data.csv",
        prediction_horizon=1
    )
    
    print(f"\n✓ Data prepared: {processed_data.shape[0]} observations, {processed_data.shape[1]} features")
    if 'zip' in processed_data.columns:
        print(f"  Zip codes: {processed_data['zip'].nunique()}")
    if 'date' in processed_data.columns:
        print(f"  Date range: {pd.to_datetime(processed_data['date']).min()} to {pd.to_datetime(processed_data['date']).max()}")
    
    # Step 2: Train models
    print("\n[2/2] Training regression models...")
    print("-" * 70)
    regressor = HousingPriceRegressor(random_state=42)
    
    results_df, importance_df = regressor.run_full_pipeline(
        data_path="data/processed_timeseries_data.csv",
        output_dir="outputs",
        is_timeseries=True,
        use_zip_holdout=True  # Uses zip code holdout for realistic validation
    )
    
    # Display results
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(results_df[['model', 'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))
    
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n✓ Complete! Check outputs/ for visualizations and results.")
    print(f"✓ Best model saved to: models/best_model.pkl")

if __name__ == "__main__":
    import pandas as pd
    main()

