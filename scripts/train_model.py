# /scripts/train_model.py

import pandas as pd
import xgboost as xgb
import joblib
import numpy as np  # Import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train():
    """
    A placeholder function to train the XGBoost model.
    This version is compatible with older scikit-learn versions.
    """
    print("Loading preprocessed data...")
    df = pd.DataFrame({
        'PM2.5': [150, 160, 155, 80, 90, 40, 50, 45],
        'NO2': [80, 85, 82, 40, 45, 20, 22, 18],
        'O3': [30, 25, 28, 60, 65, 90, 95, 92],
        'AQI': [200, 210, 205, 120, 130, 80, 90, 85]
    })

    X = df[['PM2.5', 'NO2', 'O3']]
    y = df['AQI']

    # Use a slightly larger test set for stability with small data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if X_test.empty:
        print("Test set is empty, cannot evaluate. Increase dataset size or adjust test_size.")
        return

    print("Training XGBoost model...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Evaluate model
    preds = model.predict(X_test)
    
    # --- FIX IS HERE ---
    # Calculate MSE first, then take the square root for RMSE
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    # --- END FIX ---

    print(f"Placeholder Model RMSE: {rmse}")

    print("Saving model...")
    joblib.dump(model, 'results/aqi_model.pkl')
    print("Model training complete and model saved.")

if __name__ == '__main__':
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    train()
