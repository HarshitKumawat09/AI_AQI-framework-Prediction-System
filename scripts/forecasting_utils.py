"""
Forecasting Functions for AQI Time Series Prediction
Provides date-based forecasting capabilities for the dashboard
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_forecast_model(city_name):
    """Load forecasting model for a specific city"""
    try:
        model_path = f'models/forecasting/{city_name.lower().replace(" ", "_")}_forecast.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            return None
    except Exception as e:
        print(f"Error loading model for {city_name}: {e}")
        return None

def get_available_cities():
    """Get list of cities with sufficient data for forecasting (regardless of trained models)"""
    try:
        # First try to get cities with trained models
        master_info = joblib.load('models/forecasting/master_index.pkl')
        return master_info['available_cities']
    except:
        # Fallback: return cities with sufficient historical data
        try:
            df = pd.read_csv('notebooks/aqi_dataset.csv')
            city_counts = df['City'].value_counts()
            valid_cities = city_counts[city_counts >= 100].index.tolist()
            return sorted(valid_cities)
        except:
            # Ultimate fallback: return major cities
            return ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Pune']

def forecast_city_aqi(city_name, forecast_date, confidence_interval=True):
    """
    Forecast AQI for a specific city and date

    Parameters:
    - city_name: Name of the city
    - forecast_date: datetime.date object for the forecast date
    - confidence_interval: Whether to return confidence bounds

    Returns:
    - prediction: Forecasted AQI value
    - confidence: (lower, upper) bounds if requested
    - error_message: Error message if forecast fails
    """

    # Try to load trained model first, but don't fail if not available
    model_info = load_forecast_model(city_name)
    if model_info:
        # Use trained LSTM model if available
        return forecast_with_trained_model(model_info, forecast_date, confidence_interval)
    else:
        # Use statistical forecasting approach
        return forecast_with_statistics(city_name, forecast_date, confidence_interval)

def forecast_with_statistics(city_name, forecast_date, confidence_interval=True):
    """Forecast using statistical approach with historical data"""

    try:
        df = pd.read_csv('notebooks/aqi_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        city_data = df[df['City'] == city_name].sort_values('Date')

        if len(city_data) < 30:
            return None, None, f"Insufficient historical data for {city_name} (need at least 30 records)"

        # Normalize forecast date
        forecast_date = pd.to_datetime(forecast_date).date()
        last_date = city_data['Date'].max().date()
        first_date = city_data['Date'].min().date()

        if forecast_date < first_date:
            return None, None, f"Selected date is before the earliest available data ({first_date})."

        if forecast_date <= last_date:
            # Return historical value for known dates
            record = city_data[city_data['Date'].dt.date == forecast_date]
            if not record.empty:
                observed_aqi = record['AQI'].iloc[0]
                return observed_aqi, (observed_aqi, observed_aqi), None
            # If exact date missing but within data range, continue with forecasting

        # Allow up to 2 years beyond last observed date
        days_since_last_record = (forecast_date - last_date).days
        if days_since_last_record > 730:
            return None, None, (
                f"Cannot forecast beyond 2 years after the last recorded data ({last_date}). "
                "Please choose an earlier date or extend the dataset."
            )

        # Get recent AQI values for forecasting (last 30 days)
        recent_data = city_data.tail(30)
        recent_aqi = recent_data['AQI'].values

        # Calculate basic statistics
        avg_aqi = recent_aqi.mean()
        std_aqi = recent_aqi.std()

        # Calculate trend (simple linear regression on recent data)
        from scipy import stats
        slope, intercept, _, _, _ = stats.linregress(range(len(recent_aqi)), recent_aqi)
        trend_adjustment = slope * (max(0, days_since_last_record) + 30)

        # Seasonal adjustment based on month
        city_data['month'] = city_data['Date'].dt.month
        monthly_avg = city_data.groupby('month')['AQI'].mean()
        forecast_month = forecast_date.month
        seasonal_component = monthly_avg.get(forecast_month, avg_aqi)
        seasonal_adjustment = seasonal_component - avg_aqi

        # Base prediction using recent average + trend + seasonal
        base_prediction = recent_aqi[-1]  # Start with most recent value
        prediction = base_prediction + trend_adjustment * 0.1 + seasonal_adjustment * 0.3

        # Add some noise based on historical variance
        noise = np.random.normal(0, std_aqi * 0.2)
        prediction += noise

        # Clamp to reasonable range (0-500)
        prediction = max(0, min(500, prediction))

        if confidence_interval:
            # Calculate confidence interval based on historical data and forecast horizon
            base_uncertainty = std_aqi if not np.isnan(std_aqi) else 25.0
            forecast_horizon = max(0, days_since_last_record)
            horizon_penalty = min(forecast_horizon / 730, 1.0) * base_uncertainty
            total_uncertainty = base_uncertainty + horizon_penalty

            lower_bound = max(0, prediction - total_uncertainty * 1.28)  # ~80% confidence
            upper_bound = min(500, prediction + total_uncertainty * 1.28)

            return prediction, (lower_bound, upper_bound), None
        else:
            return prediction, None, None

    except Exception as e:
        return None, None, f"Statistical forecasting error: {str(e)}"

def forecast_with_trained_model(model_info, forecast_date, confidence_interval=True):
    """Forecast using trained LSTM model (placeholder for future implementation)"""
    # For now, fall back to statistical approach
    # In the future, this would use the actual trained LSTM model
    return None, None, "LSTM forecasting not yet implemented"

def get_city_forecast_history(city_name, days_back=30):
    """Get historical AQI data for a city for context"""

    try:
        df = pd.read_csv('notebooks/aqi_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        city_data = df[df['City'] == city_name].sort_values('Date')

        # Get recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = city_data[city_data['Date'] >= cutoff_date]

        return recent_data[['Date', 'AQI']].to_dict('records')

    except Exception as e:
        print(f"Error getting history for {city_name}: {e}")
        return []

def get_forecast_accuracy_estimate(city_name, days_ahead):
    """Estimate forecast accuracy based on historical performance"""

    # This is a placeholder - in a real implementation, you'd use
    # cross-validation results from the trained models

    base_accuracy = 0.85  # 85% accuracy for short-term forecasts

    # Accuracy decreases with forecast horizon
    accuracy_decay = min(days_ahead / 730, 0.5)  # Max 50% decay over 2 years
    estimated_accuracy = base_accuracy * (1 - accuracy_decay)

    return max(0.3, estimated_accuracy)  # Minimum 30% accuracy for 2-year forecasts

# Example usage
if __name__ == "__main__":
    # Test forecasting
    cities = get_available_cities()
    if cities:
        test_city = cities[0]
        forecast_date = datetime.now().date() + timedelta(days=7)

        prediction, confidence, error = forecast_city_aqi(test_city, forecast_date)

        if prediction:
            print(f"Forecast for {test_city} on {forecast_date}:")
            print(f"AQI: {prediction:.1f}")
            if confidence:
                print(f"Confidence interval: {confidence[0]:.1f} - {confidence[1]:.1f}")
        else:
            print(f"Error: {error}")
