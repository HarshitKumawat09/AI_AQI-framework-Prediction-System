"""
COMPREHENSIVE EXPLAINABILITY VISUALIZATIONS FOR RESEARCH
Including: SHAP analysis, City patterns, Temporal analysis, Feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print("CREATING EXPLAINABILITY VISUALIZATIONS FOR RESEARCH")
print("="*80)

# Create output directory
Path('../models/explainability').mkdir(exist_ok=True, parents=True)

# Load data and model
print("\nLoading data and model...")
df = pd.read_csv('../notebooks/aqi_dataset_engineered.csv')
model = joblib.load('../models/best_model_gpu.pkl')

# Prepare features
exclude_cols = ['AQI', 'City', 'Date', 'datetime']
feature_cols = [col for col in df.columns if col not in exclude_cols and col in df.columns]
X = df[feature_cols]
y = df['AQI']

# Remove NaN
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
df_clean = df[mask].reset_index(drop=True)

print(f"✓ Dataset: {len(X)} samples, {len(feature_cols)} features")

# ============================================================================
# FIGURE 1: SHAP ANALYSIS (GLOBAL EXPLAINABILITY)
# ============================================================================
print("\n" + "="*80)
print("FIGURE 1: SHAP ANALYSIS")
print("="*80)

fig1 = plt.figure(figsize=(20, 12))

# Create SHAP explainer
print("\n[1/3] Computing SHAP values...")
# Use a sample for faster computation
sample_size = min(1000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)

# For ensemble, use the underlying estimators
if hasattr(model, 'estimators_'):
    # Use the first estimator (e.g., XGBoost)
    explainer = shap.TreeExplainer(model.estimators_[0])
    shap_values = explainer.shap_values(X_sample)
else:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

# 1.1 SHAP Summary Plot (Beeswarm)
print("[2/3] Creating SHAP summary plot...")
ax1 = plt.subplot(2, 2, 1)
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.title('SHAP Feature Importance\n(Global Impact on AQI Predictions)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# 1.2 SHAP Bar Plot (Mean absolute SHAP values)
print("[3/3] Creating SHAP bar plot...")
ax2 = plt.subplot(2, 2, 2)
shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=15)
plt.title('Feature Importance (Mean |SHAP|)\n(Higher = More Important)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# 1.3 Top 10 Features - Manual bar plot
ax3 = plt.subplot(2, 2, 3)
feature_importance = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(feature_importance)[-10:][::-1]
top_features = [X_sample.columns[i] for i in top_indices]
top_importance = feature_importance[top_indices]

colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = plt.barh(range(len(top_features)), top_importance, color=colors_feat, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Add values
for i, (bar, val) in enumerate(zip(bars, top_importance)):
    plt.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
             ha='left', va='center', fontweight='bold', fontsize=9)

# 1.4 Feature Categories Importance
ax4 = plt.subplot(2, 2, 4)
categories = {
    'Pollutants': ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2'],
    'Interactions': [f for f in X_sample.columns if any(x in f for x in ['ratio', 'product', 'interaction'])],
    'Statistical': [f for f in X_sample.columns if any(x in f for x in ['mean', 'std', 'max', 'min'])],
    'Temporal': [f for f in X_sample.columns if any(x in f for x in ['Month', 'Day', 'Weekend', 'sin', 'cos'])],
    'City': [f for f in X_sample.columns if 'City' in f],
    'Non-linear': [f for f in X_sample.columns if any(x in f for x in ['sq', 'log', 'sqrt', 'cube'])]
}

category_importance = {}
for cat, features in categories.items():
    indices = [i for i, f in enumerate(X_sample.columns) if f in features]
    if indices:
        category_importance[cat] = feature_importance[indices].sum()
    else:
        category_importance[cat] = 0

cats = list(category_importance.keys())
importances = list(category_importance.values())
colors_cat = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

bars = plt.bar(range(len(cats)), importances, color=colors_cat, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
plt.ylabel('Total SHAP Importance', fontsize=12, fontweight='bold')
plt.title('Feature Category Importance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values
for bar, val in zip(bars, importances):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../models/explainability/1_shap_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 1_shap_analysis.png")
plt.close()

# ============================================================================
# FIGURE 2: CITY-WISE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FIGURE 2: CITY-WISE PATTERNS")
print("="*80)

fig2 = plt.figure(figsize=(20, 12))

# 2.1 City Average AQI
print("\n[1/4] Creating city comparison...")
ax1 = plt.subplot(2, 3, 1)
city_aqi = df_clean.groupby('City')['AQI'].mean().sort_values(ascending=False).head(15)
colors_city = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(city_aqi)))

bars = plt.barh(range(len(city_aqi)), city_aqi.values, color=colors_city, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(city_aqi)), city_aqi.index, fontsize=10)
plt.xlabel('Average AQI', fontsize=12, fontweight='bold')
plt.title('Top 15 Most Polluted Cities\n(Average AQI)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Add AQI categories as background
for i, val in enumerate(city_aqi.values):
    color = '#d35400' if val > 200 else '#e67e22' if val > 150 else '#f39c12' if val > 100 else '#2ecc71'
    bars[i].set_color(color)
    bars[i].set_alpha(0.7)
    plt.text(val, i, f' {val:.0f}', va='center', fontweight='bold')

# 2.2 City AQI Distribution (Box plot)
print("[2/4] Creating city distribution...")
ax2 = plt.subplot(2, 3, 2)
top_cities = df_clean.groupby('City')['AQI'].mean().sort_values(ascending=False).head(8).index
city_data = [df_clean[df_clean['City'] == city]['AQI'].values for city in top_cities]

bp = plt.boxplot(city_data, labels=top_cities, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(top_cities)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('AQI', fontsize=12, fontweight='bold')
plt.title('AQI Distribution by City\n(Top 8 Cities)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# 2.3 Pollutant Profile by City
print("[3/4] Creating pollutant profiles...")
ax3 = plt.subplot(2, 3, 3)
pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']
top_3_cities = city_aqi.head(3).index

x = np.arange(len(pollutants))
width = 0.25
colors_prof = ['#e74c3c', '#3498db', '#f39c12']

for i, city in enumerate(top_3_cities):
    city_data = df_clean[df_clean['City'] == city][pollutants].mean()
    plt.bar(x + i*width, city_data, width, label=city, color=colors_prof[i], 
            alpha=0.7, edgecolor='black', linewidth=1.5)

plt.xlabel('Pollutant', fontsize=12, fontweight='bold')
plt.ylabel('Average Concentration', fontsize=12, fontweight='bold')
plt.title('Pollutant Profile\n(Top 3 Most Polluted Cities)', fontsize=14, fontweight='bold')
plt.xticks(x + width, pollutants)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 2.4 City-wise Model Performance
print("[4/4] Creating city-wise performance...")
ax4 = plt.subplot(2, 3, 4)
y_pred = model.predict(X)
df_clean['Predicted_AQI'] = y_pred

city_errors = []
city_names = []
for city in city_aqi.head(10).index:
    city_mask = df_clean['City'] == city
    city_actual = df_clean.loc[city_mask, 'AQI']
    city_pred = df_clean.loc[city_mask, 'Predicted_AQI']
    mae = np.mean(np.abs(city_actual - city_pred))
    city_errors.append(mae)
    city_names.append(city)

colors_err = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(city_errors)))
bars = plt.barh(range(len(city_names)), city_errors, color=colors_err, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(city_names)), city_names)
plt.xlabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
plt.title('Model Prediction Error by City\n(Lower is Better)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, city_errors)):
    plt.text(val, i, f' {val:.1f}', va='center', fontweight='bold', fontsize=9)

# 2.5 Geographic Heatmap (City comparison matrix)
ax5 = plt.subplot(2, 3, 5)
top_10_cities = city_aqi.head(10).index
city_pollutant_matrix = df_clean[df_clean['City'].isin(top_10_cities)].groupby('City')[['PM2.5', 'PM10', 'NO2', 'O3', 'CO']].mean()

sns.heatmap(city_pollutant_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Concentration'}, linewidths=0.5, linecolor='black')
plt.title('Pollutant Concentration Heatmap\n(Top 10 Cities)', fontsize=14, fontweight='bold')
plt.ylabel('City', fontsize=12, fontweight='bold')
plt.xlabel('Pollutant', fontsize=12, fontweight='bold')

# 2.6 City Pollution Ranking
ax6 = plt.subplot(2, 3, 6)
city_stats = df_clean.groupby('City').agg({
    'AQI': ['mean', 'max'],
    'PM2.5': 'mean'
}).round(1)
city_stats.columns = ['Avg_AQI', 'Max_AQI', 'Avg_PM2.5']
city_stats = city_stats.sort_values('Avg_AQI', ascending=False).head(10)

# Scatter plot: PM2.5 vs AQI
for i, (city, row) in enumerate(city_stats.iterrows()):
    color = '#d35400' if row['Avg_AQI'] > 200 else '#e67e22' if row['Avg_AQI'] > 150 else '#f39c12'
    plt.scatter(row['Avg_PM2.5'], row['Avg_AQI'], s=200, alpha=0.7, 
                color=color, edgecolor='black', linewidth=2)
    plt.text(row['Avg_PM2.5'], row['Avg_AQI'], city, fontsize=8, 
             ha='center', va='center', fontweight='bold')

plt.xlabel('Average PM2.5 Concentration', fontsize=12, fontweight='bold')
plt.ylabel('Average AQI', fontsize=12, fontweight='bold')
plt.title('City Pollution Clustering\n(PM2.5 vs AQI)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/explainability/2_city_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 2_city_analysis.png")
plt.close()

# ============================================================================
# FIGURE 3: TEMPORAL PATTERNS (Hourly, Monthly, Seasonal)
# ============================================================================
print("\n" + "="*80)
print("FIGURE 3: TEMPORAL PATTERNS")
print("="*80)

fig3 = plt.figure(figsize=(20, 12))

# 3.1 Monthly AQI Trend
print("\n[1/6] Creating monthly trends...")
ax1 = plt.subplot(2, 3, 1)
monthly_aqi = df_clean.groupby('Month')['AQI'].agg(['mean', 'std'])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.plot(range(1, 13), monthly_aqi['mean'], marker='o', linewidth=3, 
         markersize=10, color='#e74c3c', label='Mean AQI')
plt.fill_between(range(1, 13), 
                 monthly_aqi['mean'] - monthly_aqi['std'],
                 monthly_aqi['mean'] + monthly_aqi['std'],
                 alpha=0.3, color='#e74c3c', label='±1 Std Dev')
plt.xticks(range(1, 13), months, rotation=45)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('AQI', fontsize=12, fontweight='bold')
plt.title('Monthly AQI Variation\n(Seasonal Pattern)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Highlight winter months (high pollution)
winter_months = [11, 12, 1, 2]
for month in winter_months:
    plt.axvspan(month-0.4, month+0.4, alpha=0.1, color='blue')

# 3.2 Day of Week Pattern
print("[2/6] Creating day of week pattern...")
ax2 = plt.subplot(2, 3, 2)
dow_aqi = df_clean.groupby('DayOfWeek')['AQI'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
colors_dow = ['#3498db']*5 + ['#2ecc71']*2  # Weekdays blue, weekends green

bars = plt.bar(range(7), dow_aqi, color=colors_dow, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xticks(range(7), days)
plt.xlabel('Day of Week', fontsize=12, fontweight='bold')
plt.ylabel('Average AQI', fontsize=12, fontweight='bold')
plt.title('Day of Week AQI Pattern\n(Weekday vs Weekend)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values
for bar, val in zip(bars, dow_aqi):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 3.3 Monthly Pollutant Variations
print("[3/6] Creating pollutant seasonal patterns...")
ax3 = plt.subplot(2, 3, 3)
for pollutant in ['PM2.5', 'NO2', 'O3']:
    monthly_pol = df_clean.groupby('Month')[pollutant].mean()
    plt.plot(range(1, 13), monthly_pol, marker='o', linewidth=2, 
             markersize=8, label=pollutant)

plt.xticks(range(1, 13), months, rotation=45)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Concentration', fontsize=12, fontweight='bold')
plt.title('Seasonal Pollutant Variations', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 3.4 Weekend vs Weekday Comparison
print("[4/6] Creating weekend comparison...")
ax4 = plt.subplot(2, 3, 4)
df_clean['IsWeekend_label'] = df_clean['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
weekend_data = [
    df_clean[df_clean['IsWeekend_label'] == 'Weekday']['AQI'],
    df_clean[df_clean['IsWeekend_label'] == 'Weekend']['AQI']
]

bp = plt.boxplot(weekend_data, labels=['Weekday', 'Weekend'], patch_artist=True, showfliers=False)
colors_we = ['#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors_we):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('AQI', fontsize=12, fontweight='bold')
plt.title('Weekday vs Weekend AQI Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add mean values
means = [np.mean(data) for data in weekend_data]
positions = [1, 2]
for pos, mean in zip(positions, means):
    plt.text(pos, mean, f'μ={mean:.1f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 3.5 Seasonal Comparison
print("[5/6] Creating seasonal comparison...")
ax5 = plt.subplot(2, 3, 5)
df_clean['Season'] = df_clean['Month'].apply(
    lambda x: 'Winter' if x in [12, 1, 2] else 
              'Summer' if x in [4, 5, 6] else
              'Monsoon' if x in [7, 8, 9] else 'Autumn'
)

season_aqi = df_clean.groupby('Season')['AQI'].mean().reindex(['Winter', 'Summer', 'Monsoon', 'Autumn'])
colors_season = ['#3498db', '#f39c12', '#2ecc71', '#e67e22']

bars = plt.bar(range(4), season_aqi, color=colors_season, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xticks(range(4), season_aqi.index)
plt.ylabel('Average AQI', fontsize=12, fontweight='bold')
plt.title('Seasonal AQI Comparison\n(Indian Seasons)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values
for bar, val in zip(bars, season_aqi):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3.6 Monthly Prediction Performance
print("[6/6] Creating temporal model performance...")
ax6 = plt.subplot(2, 3, 6)
monthly_errors = df_clean.groupby('Month').apply(
    lambda x: np.mean(np.abs(x['AQI'] - x['Predicted_AQI']))
).values

plt.plot(range(1, 13), monthly_errors, marker='s', linewidth=3, 
         markersize=10, color='#9b59b6')
plt.fill_between(range(1, 13), monthly_errors, alpha=0.3, color='#9b59b6')
plt.xticks(range(1, 13), months, rotation=45)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
plt.title('Model Performance by Month\n(Prediction Error)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../models/explainability/3_temporal_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 3_temporal_analysis.png")
plt.close()

print("\n" + "="*80)
print("EXPLAINABILITY VISUALIZATION COMPLETE!")
print("="*80)
print("\n📊 Generated research-ready visualizations:")
print("   1. SHAP Analysis (Global Explainability)")
print("   2. City-wise Patterns (Geographic Analysis)")
print("   3. Temporal Patterns (Seasonal/Monthly/Weekly)")
print("\n✓ All plots saved in: models/explainability/")
print("\n🎯 Ready for research paper and presentation!")
