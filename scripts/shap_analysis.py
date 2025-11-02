# /scripts/shap_analysis.py

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def analyze():
    """
    Loads the trained model and generates SHAP explainability plots.
    """
    print("Loading the trained model and data for SHAP analysis...")

    # Load the saved model
    try:
        model = joblib.load('results/aqi_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Please run the training script first.")
        return

    # Create the same dummy data used for training to explain its predictions
    X_explain = pd.DataFrame({
        'PM2.5': [150, 160, 155, 80, 90, 40, 50, 45],
        'NO2': [80, 85, 82, 40, 45, 20, 22, 18],
        'O3': [30, 25, 28, 60, 65, 90, 95, 92]
    })

    print("Calculating SHAP values...")
    # Create a TreeExplainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # --- Generate and Save SHAP Summary Plot ---
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title('Feature Importance (SHAP Summary)')
    plt.tight_layout()
    plt.savefig('results/shap_summary_plot.png')
    plt.close()
    print("Saved 'shap_summary_plot.png' in /results folder.")

    # --- Generate and Save SHAP Bar Plot ---
    print("Generating SHAP bar plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
    plt.title('Mean SHAP Value (Overall Feature Impact)')
    plt.tight_layout()
    plt.savefig('results/shap_bar_plot.png')
    plt.close()
    print("Saved 'shap_bar_plot.png' in /results folder.")
    
    print("\nSHAP analysis complete.")

if __name__ == '__main__':
    # Ensure the results directory exists
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    analyze()
