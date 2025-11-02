"""
Comprehensive Explainability Analysis Script
Performs 4-level SHAP analysis: global, local, temporal, spatial
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path

def run_explainability():
    print("Starting comprehensive explainability analysis...")

    # Load model and data
    try:
        model = joblib.load('models/best_model_gpu.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        df = pd.read_csv('notebooks/aqi_dataset_engineered.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Prepare data using saved feature names
    X = df[feature_names]
    y = df['AQI']

    # Scale data
    X_scaled = scaler.transform(X)

    # For ensemble, use XGBoost component for SHAP
    if isinstance(model, dict):
        explainer_model = model['xgb']
    else:
        explainer_model = model

    # SHAP Explainer
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_scaled)

    # Create explainability directory
    Path('models/explainability').mkdir(exist_ok=True)

    # 1. Global Explainability - SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
    plt.savefig('models/explainability/1_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances (SHAP)')
    plt.savefig('models/explainability/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Local Explainability - Waterfall plot for high AQI
    high_aqi_idx = df['AQI'].idxmax()
    X_single = X_scaled[[high_aqi_idx]]

    shap_values_single = explainer.shap_values(X_single)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explainer.expected_value, shap_values_single[0], X_single[0], feature_names=features, show=False)
    plt.savefig('models/explainability/local_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Temporal Analysis (if Hour available)
    if 'Hour' in df.columns:
        shap_df = pd.DataFrame(shap_values, columns=features)
        shap_df['Hour'] = df['Hour'].values

        hourly_shap = shap_df.groupby('Hour').mean()

        plt.figure(figsize=(14, 8))
        hourly_shap[features[:5]].plot()  # Top 5 features
        plt.title('Average SHAP Values by Hour of Day')
        plt.ylabel('SHAP Value')
        plt.xlabel('Hour')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('models/explainability/3_temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Percentage Contribution Analysis
    print("\nPercentage Contribution Analysis for High AQI Event:")
    high_shap = shap_values[high_aqi_idx]
    total_abs_shap = np.sum(np.abs(high_shap))

    for i, feature in enumerate(features):
        percentage = (np.abs(high_shap[i]) / total_abs_shap) * 100
        print(f"{feature}: {percentage:.2f}%")

    print("Explainability analysis completed!")
    print("Visualizations saved to models/explainability/")

if __name__ == "__main__":
    run_explainability()
