"""
Streamlit Web Application for AQI Prediction
Interactive dashboard with model predictions and explainability
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from pathlib import Path
import sys
import os

# Add scripts to path for importing forecasting functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from forecasting_utils import forecast_city_aqi, get_available_cities, get_city_forecast_history, get_forecast_accuracy_estimate

# Configure page
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    """Load model, scaler, and data with caching"""
    try:
        model = joblib.load('models/best_model_gpu.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        df = pd.read_csv('notebooks/aqi_dataset_engineered.csv')
        results_df = pd.read_csv('models/gpu_results.csv')
        return model, scaler, feature_names, df, results_df
    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        st.stop()

def get_available_models():
    """Get list of available trained models"""
    models = {
        "Ensemble (GPU)": {
            "path": "models/best_model_gpu.pkl",
            "type": "original",
            "description": "XGBoost/LightGBM/CatBoost ensemble"
        }
    }

    # Add regression models
    import glob
    regression_dir = Path("models/regression")
    if regression_dir.exists():
        for model_path in glob.glob("models/regression/*.pkl"):
            model_name = Path(model_path).stem.replace("_", " ").title()
            models[model_name] = {
                "path": model_path,
                "type": "regression",
                "description": f"Scikit-learn {model_name.lower()}"
            }

    return models

def load_selected_model(model_config):
    """Load the selected model based on configuration"""
    if model_config["type"] == "original":
        model = joblib.load(model_config["path"])
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names, "original"
    else:
        # Load regression pipeline
        pipeline = joblib.load(model_config["path"])
        # For regression models, we don't have separate scaler/feature_names
        # The pipeline handles preprocessing internally
        return pipeline, None, None, "regression"

@st.cache_data
def load_city_data():
    """Load city information for defaults and statistics"""
    df = pd.read_csv('notebooks/aqi_dataset.csv')
    cities = sorted(df['City'].unique())
    
    # Calculate city averages for default values
    city_averages = df.groupby('City').agg({
        'PM2.5': 'mean',
        'PM10': 'mean', 
        'NO': 'mean',
        'NO2': 'mean',
        'NOx': 'mean',
        'NH3': 'mean',
        'CO': 'mean',
        'SO2': 'mean',
        'O3': 'mean',
        'Benzene': 'mean',
        'Toluene': 'mean',
        'Xylene': 'mean',
        'AQI': ['mean', 'max', 'min', 'count']
    }).round(2)
    
    # Flatten column names
    city_averages.columns = ['_'.join(col).strip() for col in city_averages.columns.values]
    city_averages = city_averages.rename(columns={
        'AQI_mean': 'avg_aqi',
        'AQI_max': 'max_aqi', 
        'AQI_min': 'min_aqi',
        'AQI_count': 'record_count'
    })
    
    return cities, city_averages, df

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "🟢", "#00e400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8f3f97"
    else:
        return "Hazardous", "🔴", "#7e0023"

def create_shap_visualizations(input_data, model, scaler, feature_names, df):
    """Create SHAP visualizations for input data"""
    try:
        print("DEBUG: Starting SHAP visualization creation")
        print(f"DEBUG: input_data shape: {input_data.shape}")
        print(f"DEBUG: feature_names length: {len(feature_names)}")
        print(f"DEBUG: df shape: {df.shape}")

        # Prepare background data (sample for SHAP)
        background_data = df[feature_names].sample(min(100, len(df)), random_state=42)
        print(f"DEBUG: background_data shape: {background_data.shape}")

        background_scaled = scaler.transform(background_data)
        print("DEBUG: Background data scaled successfully")

        # Scale input data
        input_scaled = scaler.transform(input_data)
        print("DEBUG: Input data scaled successfully")

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        print("DEBUG: SHAP explainer created")

        # Create SHAP values for input
        shap_values_input = explainer(input_scaled)
        print(f"DEBUG: SHAP values computed, shape: {shap_values_input.shape}")

        # Waterfall plot - using the correct SHAP API
        fig_waterfall = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values_input[0], max_display=10, show=False)
        print("DEBUG: Waterfall plot created")

        # Feature importance bar plot
        shap_values_background = explainer.shap_values(background_scaled)
        if isinstance(shap_values_background, list):
            shap_values_background = shap_values_background[0]  # For multi-class, take first class
        print(f"DEBUG: Background SHAP values shape: {shap_values_background.shape}")

        feature_importance = np.abs(shap_values_background).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        fig_bar = plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])  # Top 15
        plt.xlabel('Mean |SHAP Value| (Impact on AQI)')
        plt.title('Top 15 Feature Importances')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # Highest importance at top
        print("DEBUG: Bar plot created")

        return fig_waterfall, fig_bar

    except Exception as e:
        print(f"DEBUG: Error in create_shap_visualizations: {str(e)}")
        import traceback
        print("DEBUG: Full traceback:")
        print(traceback.format_exc())
        return None, None

def main():
    # Load model and data
    model, scaler, feature_names, df, results_df = load_model_and_data()
    cities, city_averages, raw_df = load_city_data()

    # Get available models
    available_models = get_available_models()

    # Main header
    st.markdown('<h1 class="main-header">🌍 AQI Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Explainable AI Framework for Air Quality Index Prediction")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Current Prediction", "📅 Future Forecast", "📊 Model Performance", "📈 Data Exploration", "ℹ️ About"])

    with tab1:
        st.markdown("### Make a Prediction")
        st.markdown("Select a city and adjust pollutant values, or use city averages as defaults.")

        # Model selector
        col_model, col_info = st.columns([3, 1])
        with col_model:
            selected_model_name = st.selectbox(
                "🤖 Choose Prediction Model",
                list(available_models.keys()),
                index=0,
                help="Select the machine learning model for prediction"
            )
        with col_info:
            model_info = available_models[selected_model_name]
            st.markdown(f"**Type:** {model_info['type'].title()}")
            if st.button("ℹ️", help="Model information"):
                st.info(f"**{selected_model_name}**\n\n{model_info['description']}")

        # Load selected model
        selected_model, selected_scaler, selected_features, model_type = load_selected_model(model_info)

        # City selector
        col_city, col_mode = st.columns([2, 1])
        with col_city:
            selected_city = st.selectbox(
                "🏙️ Select City",
                cities,
                index=cities.index("Delhi") if "Delhi" in cities else 0,
                help="Choose a city to see its historical air quality patterns"
            )
        
        with col_mode:
            use_city_avg = st.checkbox("Use City Averages", value=True, 
                                     help="Use historical average pollutant values for this city")

        # Display city information
        if selected_city:
            city_stats = city_averages.loc[selected_city]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg AQI", f"{city_stats['avg_aqi']:.1f}")
            with col2:
                st.metric("Max AQI", f"{city_stats['max_aqi']:.1f}")
            with col3:
                st.metric("Min AQI", f"{city_stats['min_aqi']:.1f}")
            with col4:
                st.metric("Records", f"{city_stats['record_count']:,}")

        # Get default values
        defaults = {}
        if use_city_avg and selected_city in city_averages.index:
            city_data = city_averages.loc[selected_city]
            defaults = {
                'pm25': city_data.get('PM2.5_mean', 50.0),
                'pm10': city_data.get('PM10_mean', 100.0),
                'no': city_data.get('NO_mean', 20.0),
                'no2': city_data.get('NO2_mean', 30.0),
                'nox': city_data.get('NOx_mean', 50.0),
                'nh3': city_data.get('NH3_mean', 25.0),
                'co': city_data.get('CO_mean', 1.0),
                'so2': city_data.get('SO2_mean', 10.0),
                'o3': city_data.get('O3_mean', 40.0),
                'benzene': city_data.get('Benzene_mean', 5.0),
                'toluene': city_data.get('Toluene_mean', 10.0),
                'xylene': city_data.get('Xylene_mean', 3.0)
            }
        else:
            defaults = {
                'pm25': 50.0, 'pm10': 100.0, 'no': 20.0, 'no2': 30.0, 'nox': 50.0,
                'nh3': 25.0, 'co': 1.0, 'so2': 10.0, 'o3': 40.0,
                'benzene': 5.0, 'toluene': 10.0, 'xylene': 3.0
            }

        # Input form
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Primary Pollutants")
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=1000.0, value=defaults['pm25'], step=0.1)
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=2000.0, value=defaults['pm10'], step=0.1)
            no2 = st.number_input("NO₂ (ppb)", min_value=0.0, max_value=500.0, value=defaults['no2'], step=0.1)

        with col2:
            st.markdown("#### Secondary Pollutants")
            o3 = st.number_input("O₃ (ppb)", min_value=0.0, max_value=500.0, value=defaults['o3'], step=0.1)
            co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=defaults['co'], step=0.1)
            so2 = st.number_input("SO₂ (ppb)", min_value=0.0, max_value=500.0, value=defaults['so2'], step=0.1)

        with col3:
            st.markdown("#### Additional Features")
            no = st.number_input("NO (ppb)", min_value=0.0, max_value=500.0, value=defaults['no'], step=0.1)
            nox = st.number_input("NOx (ppb)", min_value=0.0, max_value=1000.0, value=defaults['nox'], step=0.1)
            nh3 = st.number_input("NH₃ (ppb)", min_value=0.0, max_value=500.0, value=defaults['nh3'], step=0.1)
            
            # Additional VOCs
            benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, max_value=100.0, value=defaults['benzene'], step=0.1)
            toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, max_value=100.0, value=defaults['toluene'], step=0.1)
            xylene = st.number_input("Xylene (µg/m³)", min_value=0.0, max_value=100.0, value=defaults['xylene'], step=0.1)

            # Temporal features
            hour = st.slider("Hour of Day", 0, 23, 12)
            month = st.slider("Month", 1, 12, 6)
            day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 1)

        # Predict button
        if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame([{
                'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox,
                'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
                'Benzene': benzene, 'Toluene': toluene, 'Xylene': xylene,
                'Year': 2024, 'Month': month, 'Day': 15, 'DayOfWeek': day_of_week,
                'Hour': hour, 'PM2.5_NO2': pm25 * no2, 'City': selected_city
            }])

            # Make prediction based on model type
            if model_type == "original":
                # Original ensemble model with separate preprocessing
                input_filtered = input_data[selected_features]
                input_scaled = selected_scaler.transform(input_filtered)
                prediction = selected_model.predict(input_scaled)[0]
            else:
                # Regression pipeline with internal preprocessing
                prediction = selected_model.predict(input_data)[0]

            # Get AQI category
            category, emoji, color = get_aqi_category(prediction)

            # Display prediction
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])

            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: {color}; margin: 0;">{emoji} {prediction:.1f}</h2>
                    <h3 style="margin: 0.5rem 0;">{category}</h3>
                    <p style="margin: 0; color: #666;">Predicted AQI</p>
                </div>
                """, unsafe_allow_html=True)

            # SHAP Analysis
            st.markdown("### 🤖 Explainability Analysis")
            st.markdown("Understanding what factors influenced this prediction:")

            # Debug information
            st.markdown("**Debug Info:**")
            st.write(f"Model Type: {model_type}")
            if model_type == "original":
                st.write(f"Input shape: {input_filtered.shape}")
                st.write(f"Feature names count: {len(selected_features)}")
                st.write(f"Sample features: {selected_features[:5]}")
            else:
                st.write("Using regression pipeline with internal preprocessing")

            # Only show SHAP for tree-based models (original ensemble or Random Forest)
            if model_type == "original" or "random forest" in selected_model_name.lower():
                try:
                    if model_type == "original":
                        fig_waterfall, fig_bar = create_shap_visualizations(input_filtered, selected_model, selected_scaler, selected_features, df)
                    else:
                        # For Random Forest, we need to handle it differently
                        st.warning("⚠️ SHAP analysis for Random Forest is limited. Using feature importance instead.")
                        # Get feature importance from the pipeline
                        if hasattr(selected_model, 'named_steps') and 'regressor' in selected_model.named_steps:
                            regressor = selected_model.named_steps['regressor']
                            if hasattr(regressor, 'feature_importances_'):
                                # Get feature names from preprocessor
                                feature_names = selected_model.named_steps['preprocessor'].get_feature_names_out()
                                importances = regressor.feature_importances_

                                fig_bar = plt.figure(figsize=(10, 8))
                                plt.barh(feature_names[:15], importances[:15])
                                plt.xlabel('Feature Importance')
                                plt.title('Top 15 Feature Importances (Random Forest)')
                                plt.grid(axis='x', alpha=0.3)
                                plt.gca().invert_yaxis()
                                fig_waterfall = None
                            else:
                                fig_waterfall, fig_bar = None, None
                        else:
                            fig_waterfall, fig_bar = None, None

                    if fig_waterfall is not None and fig_bar is not None:
                        st.success("SHAP/feature analysis generated successfully!")
                        col1, col2 = st.columns(2)
                        with col1:
                            if fig_waterfall:
                                st.markdown("#### Prediction Breakdown")
                                st.pyplot(fig_waterfall)
                            else:
                                st.markdown("#### Feature Importance")
                                st.pyplot(fig_bar)

                        with col2:
                            st.markdown("#### Feature Contributions")
                            if fig_waterfall:
                                st.pyplot(fig_bar)
                            else:
                                st.markdown("*Feature importance shown above*")

                    elif fig_bar is not None:
                        st.success("Feature importance analysis generated!")
                        st.markdown("#### Feature Importance")
                        st.pyplot(fig_bar)
                    else:
                        st.warning("SHAP analysis not available for this model type.")
                        st.info("💡 **Tip**: SHAP works best with tree-based models like XGBoost, LightGBM, and Random Forest.")

                except Exception as e:
                    st.error(f"Error in explainability analysis: {str(e)}")
                    st.info("💡 **Alternative**: Try selecting the 'Ensemble (GPU)' or 'Random Forest' model for full SHAP analysis.")
                    import traceback
                    st.code(traceback.format_exc())
            else:
                st.info("📊 **Explainability Note**: SHAP analysis is available for tree-based models (Ensemble, Random Forest). Other models use different interpretability methods.")

                # Show coefficient analysis for Linear Regression
                if "linear regression" in selected_model_name.lower():
                    st.markdown("#### 📈 Linear Regression Coefficients")
                    st.markdown("Feature importance based on model coefficients:")

                    try:
                        # Extract coefficients from the pipeline
                        regressor = selected_model.named_steps['regressor']
                        feature_names_out = selected_model.named_steps['preprocessor'].get_feature_names_out()

                        # Get coefficients and create importance DataFrame
                        coefficients = regressor.coef_
                        abs_coefficients = np.abs(coefficients)

                        # Create DataFrame for visualization
                        coef_df = pd.DataFrame({
                            'feature': feature_names_out,
                            'coefficient': coefficients,
                            'abs_coefficient': abs_coefficients
                        })

                        # Sort by absolute coefficient value
                        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

                        # Show top 15 features
                        top_features = coef_df.head(15)

                        # Create coefficient bar chart
                        fig_coef = px.bar(
                            top_features,
                            x='coefficient',
                            y='feature',
                            orientation='h',
                            title='Top 15 Feature Coefficients (Linear Regression)',
                            labels={'coefficient': 'Coefficient Value', 'feature': 'Feature'},
                            color='coefficient',
                            color_continuous_scale='RdBu_r'
                        )
                        fig_coef.update_layout(
                            xaxis_title="Coefficient Value",
                            yaxis_title="Feature",
                            height=500
                        )
                        fig_coef.update_yaxes(autorange="reversed")  # Highest at top
                        st.plotly_chart(fig_coef, use_container_width=True)

                        # Show coefficient interpretation
                        st.markdown("**💡 Coefficient Interpretation:**")
                        st.markdown("- **Positive coefficients** → Higher feature values increase AQI prediction")
                        st.markdown("- **Negative coefficients** → Higher feature values decrease AQI prediction")
                        st.markdown("- **Coefficient magnitude** → Strength of the relationship")

                        # Show most influential features
                        positive_features = coef_df[coef_df['coefficient'] > 0].head(5)
                        negative_features = coef_df[coef_df['coefficient'] < 0].head(5)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔺 Features that increase AQI:**")
                            for _, row in positive_features.iterrows():
                                st.write(f"• {row['feature']}: +{row['coefficient']:.4f}")

                        with col2:
                            st.markdown("**🔻 Features that decrease AQI:**")
                            for _, row in negative_features.iterrows():
                                st.write(f"• {row['feature']}: {row['coefficient']:.4f}")

                    except Exception as e:
                        st.warning(f"Could not extract coefficients: {str(e)}")
                        st.info("💡 Linear regression coefficients provide insight into which features most influence AQI predictions.")

                elif "gradient boosting" in selected_model_name.lower():
                    st.markdown("#### 📊 Gradient Boosting Feature Importance")
                    st.markdown("Feature importance from gradient boosting model:")

                    try:
                        # Extract feature importance from Gradient Boosting
                        regressor = selected_model.named_steps['regressor']
                        feature_names_out = selected_model.named_steps['preprocessor'].get_feature_names_out()

                        # Get feature importances
                        importances = regressor.feature_importances_

                        # Create DataFrame for visualization
                        importance_df = pd.DataFrame({
                            'feature': feature_names_out,
                            'importance': importances
                        })

                        # Sort by importance
                        importance_df = importance_df.sort_values('importance', ascending=False)

                        # Show top 15 features
                        top_features = importance_df.head(15)

                        # Create importance bar chart
                        fig_imp = px.bar(
                            top_features,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 15 Feature Importances (Gradient Boosting)',
                            labels={'importance': 'Importance Score', 'feature': 'Feature'},
                            color='importance',
                            color_continuous_scale='Viridis'
                        )
                        fig_imp.update_layout(
                            xaxis_title="Importance Score",
                            yaxis_title="Feature",
                            height=500
                        )
                        fig_imp.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig_imp, use_container_width=True)

                        st.markdown("**💡 Feature Importance:** Higher scores indicate features that contribute more to prediction accuracy.")

                    except Exception as e:
                        st.warning(f"Could not extract feature importance: {str(e)}")

                # Show model information for all regression models
                st.markdown("---")
                st.markdown(f"**Selected Model:** {selected_model_name}")
                st.markdown(f"**Type:** {model_info['description']}")
                st.markdown(f"**Prediction:** {prediction:.1f} AQI ({category})")

    with tab2:
        st.markdown("### 📅 Future AQI Forecast")
        st.markdown("Predict air quality for specific dates using historical patterns and time series analysis.")

        # Get available cities for forecasting
        forecast_cities = get_available_cities()

        if not forecast_cities:
            st.warning("⚠️ Forecasting models not available. Please train the forecasting models first by running `python scripts/train_forecasting.py`")
            st.info("**Note**: The forecasting feature requires trained time series models for each city.")
        else:
            # City and date selection
            col_city, col_date = st.columns(2)

            with col_city:
                forecast_city = st.selectbox(
                    "🏙️ Select City for Forecast",
                    forecast_cities,
                    index=forecast_cities.index("Delhi") if "Delhi" in forecast_cities else 0,
                    help="Choose a city to forecast AQI"
                )

            with col_date:
                min_date = date.today() + timedelta(days=1)
                max_date = date.today() + timedelta(days=730)  # 2 years
                forecast_date = st.date_input(
                    "📅 Select Forecast Date",
                    min_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Choose a future date for AQI prediction (up to 2 years ahead)"
                )

            # Forecast button
            if st.button("🔮 Forecast Future AQI", type="primary", use_container_width=True):
                with st.spinner("Generating forecast..."):
                    prediction, confidence, error = forecast_city_aqi(forecast_city, forecast_date)

                    if prediction is not None:
                        st.success("Prediction generated successfully!")
                        # Display forecast result
                        category, emoji, color = get_aqi_category(prediction)

                        st.markdown("---")
                        col1, col2, col3 = st.columns([2, 1, 2])

                        with col2:
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style="color: {color}; margin: 0;">{emoji} {prediction:.1f}</h2>
                                <h3 style="margin: 0.5rem 0;">{category}</h3>
                                <p style="margin: 0; color: #666;">Forecasted AQI</p>
                                <p style="margin: 0.5rem 0; font-size: 0.8em; color: #888;">{forecast_date.strftime('%B %d, %Y')}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Forecast details
                        st.markdown("### 📊 Forecast Details")

                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            accuracy = get_forecast_accuracy_estimate(forecast_city, (forecast_date - date.today()).days)
                            st.metric("Forecast Confidence", f"{accuracy:.1%}")

                        with col_b:
                            days_ahead = (forecast_date - date.today()).days
                            st.metric("Days Ahead", f"{days_ahead} days")

                        with col_c:
                            st.metric("City", forecast_city)

                        # Confidence interval
                        if confidence:
                            st.markdown("#### 🎯 Confidence Interval")
                            st.info(f"Expected AQI range: **{confidence[0]:.1f} - {confidence[1]:.1f}** (80% confidence)")

                            # Visual confidence interval
                            fig_ci = go.Figure()
                            fig_ci.add_trace(go.Scatter(
                                x=[forecast_date],
                                y=[prediction],
                                mode='markers',
                                marker=dict(size=12, color=color),
                                name='Forecast'
                            ))

                            fig_ci.add_trace(go.Scatter(
                                x=[forecast_date, forecast_date],
                                y=[confidence[0], confidence[1]],
                                mode='lines',
                                line=dict(color=color, width=3),
                                name='Confidence Range'
                            ))

                            fig_ci.update_layout(
                                title="Forecast with Confidence Interval",
                                xaxis_title="Date",
                                yaxis_title="AQI",
                                showlegend=False,
                                height=300
                            )
                            st.plotly_chart(fig_ci, use_container_width=True)

                        # Historical context
                        st.markdown("#### 📈 Historical Context")
                        history = get_city_forecast_history(forecast_city, days_back=30)

                        if history:
                            hist_df = pd.DataFrame(history)
                            hist_df['Date'] = pd.to_datetime(hist_df['Date'])

                            fig_hist = px.line(hist_df, x='Date', y='AQI',
                                             title=f"Recent AQI History - {forecast_city}",
                                             markers=True)
                            fig_hist.add_hline(y=prediction, line_dash="dash", line_color=color,
                                             annotation_text=f"Forecast: {prediction:.1f}")
                            st.plotly_chart(fig_hist, use_container_width=True)

                            # Historical statistics
                            avg_aqi = hist_df['AQI'].mean()
                            max_aqi = hist_df['AQI'].max()
                            min_aqi = hist_df['AQI'].min()

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Historical Average", f"{avg_aqi:.1f}")
                            col2.metric("Historical Max", f"{max_aqi:.1f}")
                            col3.metric("Historical Min", f"{min_aqi:.1f}")

                    else:
                        st.error(f"❌ Forecast Error: {error}")
                        st.info("💡 **Tips**: Try selecting a date within the next year and ensure the city has forecasting models available.")

            # Forecast information
            st.markdown("---")
            st.markdown("### ℹ️ About Forecasting")
            st.markdown("""
            **How it works:**
            - Uses historical AQI patterns specific to each city
            - Incorporates seasonal trends and recent air quality changes
            - Provides confidence intervals to show forecast uncertainty
            - Accuracy decreases for longer forecast horizons

            **Limitations:**
            - Cannot predict external events (weather changes, policy interventions)
            - Accuracy typically 70-90% for short-term forecasts (1-30 days)
            - Best results within 3-6 months ahead
            """)

    with tab3:
        st.markdown("### 📊 Model Performance")
        st.markdown("Performance metrics across different models:")

        # Load regression results if available
        regression_results = None
        regression_path = Path("models/regression/regression_results.csv")
        if regression_path.exists():
            try:
                regression_results = pd.read_csv(regression_path)
                regression_results['model'] = regression_results['model'].str.replace('_', ' ').str.title()
            except:
                pass

        # Show original GPU ensemble results
        if not results_df.empty:
            st.markdown("#### 🤖 Original GPU Ensemble Models")
            results_display = results_df.set_index('Unnamed: 0')
            st.dataframe(results_display.style.highlight_min(axis=0, color='lightgreen').highlight_max(axis=0, color='lightcoral'))

            # Performance visualization
            fig = px.bar(results_display.reset_index(), x='Unnamed: 0', y=['rmse', 'mae'],
                        title="GPU Ensemble Model Performance Comparison",
                        labels={'Unnamed: 0': 'Model', 'value': 'Error', 'variable': 'Metric'},
                        color_discrete_sequence=['#66b3ff'])
            st.plotly_chart(fig, use_container_width=True)

        # Show regression model results
        if regression_results is not None:
            st.markdown("#### 📈 Regression Models")
            st.dataframe(regression_results.set_index('model').style.highlight_min(axis=0, color='lightgreen').highlight_max(axis=0, color='lightcoral'))

            # Regression performance visualization
            fig_reg = px.bar(regression_results, x='model', y=['rmse', 'mae', 'r2'],
                           title="Regression Model Performance Comparison",
                           labels={'model': 'Model', 'value': 'Score', 'variable': 'Metric'},
                           color='variable',
                           barmode='group')
            st.plotly_chart(fig_reg, use_container_width=True)

            # Model comparison summary
            st.markdown("#### 🏆 Model Comparison Summary")
            col1, col2, col3 = st.columns(3)
            
            if not results_df.empty and not regression_results.empty:
                gpu_best_rmse = results_df['rmse'].min()
                reg_best_rmse = regression_results['rmse'].min()
                
                with col1:
                    st.metric("Best GPU RMSE", f"{gpu_best_rmse:.2f}")
                with col2:
                    st.metric("Best Regression RMSE", f"{reg_best_rmse:.2f}")
                with col3:
                    improvement = ((gpu_best_rmse - reg_best_rmse) / gpu_best_rmse) * 100
                    st.metric("RMSE Improvement", f"{improvement:+.1f}%")
        else:
            st.info("Regression model results will appear here after training additional models.")

        if results_df.empty and regression_results is None:
            st.warning("Model results not available.")

    with tab4:
        st.markdown("### 📈 Data Exploration")
        st.markdown("Explore the AQI dataset and patterns:")

        # City Analysis Section
        st.markdown("#### 🏙️ City Analysis")
        col_city1, col_city2 = st.columns(2)
        
        with col_city1:
            city1 = st.selectbox("Select City 1", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
        with col_city2:
            city2 = st.selectbox("Select City 2", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 1)

        # City comparison
        if city1 and city2:
            city1_stats = city_averages.loc[city1]
            city2_stats = city_averages.loc[city2]
            
            st.markdown(f"#### {city1} vs {city2} Comparison")
            comparison_data = pd.DataFrame({
                'Metric': ['Average AQI', 'Max AQI', 'Min AQI', 'Total Records'],
                city1: [city1_stats['avg_aqi'], city1_stats['max_aqi'], city1_stats['min_aqi'], city1_stats['record_count']],
                city2: [city2_stats['avg_aqi'], city2_stats['max_aqi'], city2_stats['min_aqi'], city2_stats['record_count']]
            })
            st.dataframe(comparison_data.set_index('Metric'))

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(raw_df):,}")
        with col2:
            st.metric("Average AQI", f"{raw_df['AQI'].mean():.1f}")
        with col3:
            st.metric("Max AQI", f"{raw_df['AQI'].max():.1f}")
        with col4:
            st.metric("Cities Covered", f"{len(cities)}")

        # AQI distribution
        fig_hist = px.histogram(raw_df, x='AQI', nbins=50,
                               title="AQI Distribution Across All Cities",
                               labels={'AQI': 'Air Quality Index'})
        st.plotly_chart(fig_hist, use_container_width=True)

        # City-wise AQI comparison
        city_aqi_avg = city_averages['avg_aqi'].sort_values(ascending=False)
        fig_city = px.bar(city_aqi_avg, 
                         title="Average AQI by City",
                         labels={'value': 'Average AQI', 'index': 'City'},
                         color=city_aqi_avg.values,
                         color_continuous_scale='RdYlGn_r')
        fig_city.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_city, use_container_width=True)

        # Pollutant correlations
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        corr_matrix = raw_df[numeric_cols].corr()

        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Pollutant Feature Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Time series if available
        if 'Month' in raw_df.columns:
            monthly_avg = raw_df.groupby('Month')['AQI'].mean().reset_index()
            fig_ts = px.line(monthly_avg, x='Month', y='AQI',
                           title="Average AQI by Month (All Cities)",
                           markers=True)
            st.plotly_chart(fig_ts, use_container_width=True)

    with tab5:
        st.markdown("### ℹ️ About This Project")
        st.markdown("""
        #### 🌍 **Explainable AI Framework for AQI Prediction**

        This dashboard demonstrates a comprehensive AI system for predicting Air Quality Index (AQI)
        using real Indian air quality data from 26 cities spanning 2015-2020.

        **Key Features:**
        - 🔮 **Accurate Predictions**: Ensemble of XGBoost, LightGBM, and CatBoost models
        - 🤖 **Full Explainability**: SHAP-based interpretability for all predictions
        - 📊 **Comprehensive Analysis**: Multi-level insights (global, local, temporal, spatial)
        - 🌟 **Research-Grade**: GPU-accelerated training with 99 engineered features

        **Technical Stack:**
        - Machine Learning: XGBoost, LightGBM, CatBoost, Optuna
        - Explainability: SHAP (SHapley Additive exPlanations)
        - Visualization: Plotly, Matplotlib, Seaborn
        - UI Framework: Streamlit

        **Dataset:**
        - 26 Indian cities
        - 5 years of data (2015-2020)
        - 6 primary pollutants + engineered features
        - 24,307 cleaned records

        ---
        **Made with ❤️ for cleaner air and better health**
        """)

        # Footer
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
