"""
Classical Machine Learning Model Benchmarking
Trains and evaluates XGBoost, CatBoost, LightGBM models
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from pathlib import Path

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost"""
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'tree_method': 'hist'  # Use CPU
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def train_models():
    print("Starting classical ML model benchmarking...")

    # Load engineered data
    try:
        df = pd.read_csv('notebooks/aqi_dataset_engineered.csv')
        print(f"Loaded engineered dataset with {len(df)} records")
    except FileNotFoundError:
        print("Error: notebooks/aqi_dataset_engineered.csv not found. Run data preprocessing first.")
        return

    # Define features (use numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'AQI' in numeric_cols:
        numeric_cols.remove('AQI')
    
    print(f"Using {len(numeric_cols)} numeric features")
    print(f"Sample features: {numeric_cols[:10]}")
    
    X = df[numeric_cols]
    y = df['AQI']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # XGBoost with Optuna
    print("Training XGBoost with GPU acceleration...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=15)

    best_xgb = xgb.XGBRegressor(**study_xgb.best_params)
    best_xgb.fit(X_train_scaled, y_train)
    xgb_preds = best_xgb.predict(X_test_scaled)
    results['XGBoost'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, xgb_preds)),
        'mae': mean_absolute_error(y_test, xgb_preds),
        'r2': r2_score(y_test, xgb_preds)
    }

    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=6)
    lgb_model.fit(X_train_scaled, y_train)
    lgb_preds = lgb_model.predict(X_test_scaled)
    results['LightGBM'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, lgb_preds)),
        'mae': mean_absolute_error(y_test, lgb_preds),
        'r2': r2_score(y_test, lgb_preds)
    }

    # CatBoost
    print("Training CatBoost...")
    cb_model = cb.CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=False)
    cb_model.fit(X_train_scaled, y_train)
    cb_preds = cb_model.predict(X_test_scaled)
    results['CatBoost'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, cb_preds)),
        'mae': mean_absolute_error(y_test, cb_preds),
        'r2': r2_score(y_test, cb_preds)
    }

    # Ensemble (simple average)
    ensemble_preds = (xgb_preds + lgb_preds + cb_preds) / 3
    results['Ensemble'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        'mae': mean_absolute_error(y_test, ensemble_preds),
        'r2': r2_score(y_test, ensemble_preds)
    }

    # Save best model (XGBoost for SHAP)
    joblib.dump(best_xgb, 'models/best_model_gpu.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature names
    import json
    with open('models/feature_names.json', 'w') as f:
        json.dump(numeric_cols, f)

    # Save ensemble separately
    ensemble_model = {
        'xgb': best_xgb,
        'lgb': lgb_model,
        'cb': cb_model,
        'scaler': scaler
    }
    joblib.dump(ensemble_model, 'models/ensemble_model.pkl')

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/gpu_results.csv')

    print("Model benchmarking completed!")
    print("\nResults:")
    print(results_df)

    return results

if __name__ == "__main__":
    train_models()
