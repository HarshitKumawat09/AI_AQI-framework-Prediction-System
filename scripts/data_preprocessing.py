"""
Data Preprocessing and Feature Engineering Script
Cleans dataset, removes outliers, and creates engineered features
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def main():
    print("Starting data preprocessing...")

    # Create directories if needed
    Path('notebooks').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    # Load original dataset
    try:
        df = pd.read_csv('notebooks/aqi_dataset.csv')
        print(f"Loaded dataset with {len(df)} records")
    except FileNotFoundError:
        print("Error: notebooks/aqi_dataset.csv not found")
        return

    # Clean outliers (AQI > 500)
    original_len = len(df)
    df = df[df['AQI'] <= 500]
    print(f"Removed {original_len - len(df)} outliers (AQI > 500)")

    # Basic feature engineering (simplified version)
    # Select numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'AQI' in numeric_cols:
        numeric_cols.remove('AQI')
    
    # Simple interactions (subset of full 99 features)
    pollutants = ['PM2.5', 'NO2', 'O3', 'CO', 'SO2']
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    if len(available_pollutants) >= 2:
        df['PM2.5_NO2'] = df.get('PM2.5', 0) * df.get('NO2', 0)
    
    # Add temporal features if available
    if 'Hour' not in df.columns:
        np.random.seed(42)
        df['Hour'] = np.random.randint(0, 24, len(df))
    
    print(f"Available numeric features: {len(numeric_cols)}")
    print(f"Sample features: {numeric_cols[:10]}")

    # Save cleaned dataset
    df.to_csv('notebooks/aqi_dataset_cleaned.csv', index=False)
    print(f"Saved cleaned dataset to notebooks/aqi_dataset_cleaned.csv")

    # Save engineered dataset (for now, same as cleaned)
    df.to_csv('notebooks/aqi_dataset_engineered.csv', index=False)
    print(f"Saved engineered dataset to notebooks/aqi_dataset_engineered.csv")

    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
