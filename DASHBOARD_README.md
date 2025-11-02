# 🌍 AQI Prediction Dashboard

## How to Run

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the dashboard**:
   - Open your browser to `http://localhost:8501`
   - Or use the browser preview button in your IDE

## Features

- 🔮 **Interactive AQI Prediction**: Input pollutant values and get instant predictions
- 🏙️ **City-Based Prediction**: Select any of 26 Indian cities and use historical averages as defaults
- 📅 **Future AQI Forecasting**: Predict air quality for specific future dates using time series analysis
- 🤖 **Explainability**: SHAP-based analysis showing what factors influenced the prediction (waterfall plots + feature importance)
- 📊 **Model Performance**: Compare different ML models (XGBoost, LightGBM, CatBoost)
- 📈 **Data Exploration**: Visualize AQI distributions, correlations, and temporal patterns
- 🏆 **City Comparison**: Compare air quality statistics between different cities
- 🎨 **Beautiful UI**: Modern, responsive design with custom styling

## Dashboard Sections

1. **Current Prediction Tab**: Main interface for making predictions
   - City selector with 26 Indian cities
   - Use city averages as default values or custom inputs
   - Real-time AQI prediction with category indicators
   - SHAP explainability visualizations

2. **Future Forecast Tab**: Time series forecasting for future dates
   - Date picker for forecasting (up to 1 year ahead)
   - Confidence intervals and uncertainty estimates
   - Historical context and trend analysis
   - Forecast accuracy indicators

3. **Model Performance**: View and compare model metrics
4. **Data Exploration**: Explore the dataset and patterns
   - City comparison tools
   - AQI distribution across cities
   - Pollutant correlation analysis
   - Monthly trends
5. **About**: Project information and technical details

## Forecasting Capabilities

**Available Cities**: Ahmedabad, Aizawl, Amaravati, Amritsar, Bengaluru, Bhopal, Brajrajnagar, Chandigarh, Chennai, Coimbatore, Delhi, Ernakulam, Gurugram, Guwahati, Hyderabad, Jaipur, Jorapokhar, Kochi, Kolkata, Lucknow, Mumbai, Patna, Shillong, Talcher, Thiruvananthapuram, Visakhapatnam

**Forecast Features**:
- **Date-Specific Predictions**: Forecast AQI for any future date
- **Confidence Intervals**: 80% confidence ranges for forecast uncertainty
- **Historical Context**: Recent AQI trends for context
- **Accuracy Estimates**: Forecast reliability based on time horizon
- **Seasonal Adjustments**: Accounts for monthly and seasonal patterns

**Forecast Range**: 1 day to 2 years ahead from current date

## Input Features

The model uses these features for prediction:
- Primary pollutants: PM2.5, PM10, NO₂, O₃, CO, SO₂
- Additional: NO, NOx, NH₃
- VOCs: Benzene, Toluene, Xylene (Volatile Organic Compounds)
- Temporal: Hour, Month, Day of Week
- Engineered: PM2.5 × NO₂ interaction

## Output

- **AQI Prediction**: Numerical value with category (Good, Moderate, etc.)
- **SHAP Analysis**: Waterfall plot and feature contribution chart
- **Color-coded Results**: Visual indicators for air quality levels
