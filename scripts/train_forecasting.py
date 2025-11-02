"""
Time Series Forecasting for AQI Prediction
Trains LSTM models for each city to forecast future AQI values
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster(nn.Module):
    """LSTM model for time series forecasting"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_city_forecast_model(city_data, city_name, seq_length=30, epochs=100):
    """Train LSTM model for a specific city"""

    # Prepare time series data
    city_data = city_data.sort_values('Date')
    aqi_values = city_data['AQI'].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(aqi_values)

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    if len(X) == 0:
        print(f"Not enough data for {city_name} (need at least {seq_length+1} records)")
        return None, None

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'{city_name} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs.squeeze(), y_test.squeeze())
        rmse = np.sqrt(test_loss.item())

    print(f'{city_name} - Test RMSE: {rmse:.4f}')

    # Save model and scaler
    model_info = {
        'model': model,
        'scaler': scaler,
        'seq_length': seq_length,
        'city': city_name,
        'rmse': rmse,
        'last_date': city_data['Date'].max()
    }

    return model_info

def forecast_aqi(model_info, forecast_date, confidence_interval=True):
    """Forecast AQI for a specific date"""

    model = model_info['model']
    scaler = model_info['scaler']
    seq_length = model_info['seq_length']
    last_date = model_info['last_date']

    # Calculate days to forecast
    days_ahead = (forecast_date - last_date).days

    if days_ahead <= 0:
        return None, "Forecast date must be after the last training date"

    if days_ahead > 365:  # Limit to 1 year ahead
        return None, "Cannot forecast more than 1 year ahead"

    # For simplicity, use the last known sequence to forecast
    # In a full implementation, you'd use a rolling forecast
    model.eval()

    # Generate forecast (simplified - using last sequence)
    with torch.no_grad():
        # This is a placeholder - actual implementation would need historical data
        # For demo purposes, we'll return a simulated forecast
        base_prediction = np.random.normal(150, 50)  # Simulated prediction
        prediction = max(0, min(500, base_prediction))  # Clamp to reasonable range

    if confidence_interval:
        # Simple confidence interval (placeholder)
        lower_bound = max(0, prediction - 30)
        upper_bound = min(500, prediction + 30)
        return prediction, (lower_bound, upper_bound)
    else:
        return prediction, None

def train_all_city_models():
    """Train forecasting models for all cities with sufficient data"""

    print("Loading data...")
    df = pd.read_csv('notebooks/aqi_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter cities with enough data (at least 100 records)
    city_counts = df['City'].value_counts()
    valid_cities = city_counts[city_counts >= 100].index.tolist()

    print(f"Training models for {len(valid_cities)} cities with sufficient data")

    os.makedirs('models/forecasting', exist_ok=True)
    city_models = {}

    for city in valid_cities:
        print(f"\nTraining model for {city}...")
        city_data = df[df['City'] == city].copy()

        model_info = train_city_forecast_model(city_data, city)

        if model_info:
            # Save individual model
            model_path = f'models/forecasting/{city.lower().replace(" ", "_")}_forecast.pkl'
            joblib.dump(model_info, model_path)
            city_models[city] = model_info
            print(f"Model saved for {city}")
        else:
            print(f"Skipped {city} - insufficient data")

    # Save master index
    master_info = {
        'available_cities': list(city_models.keys()),
        'last_updated': datetime.now(),
        'model_version': '1.0'
    }
    joblib.dump(master_info, 'models/forecasting/master_index.pkl')

    print(f"\nTraining complete! Models available for {len(city_models)} cities:")
    for city in sorted(city_models.keys()):
        print(f"  - {city}")

    return city_models

if __name__ == "__main__":
    train_all_city_models()
