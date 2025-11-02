"""
Comprehensive Visualization of Model Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

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

# Make predictions
print("Generating predictions...")
y_pred = model.predict(X)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"✓ RMSE: {rmse:.2f}")
print(f"✓ MAE: {mae:.2f}")
print(f"✓ R²: {r2:.4f}")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# 1. ACTUAL VS PREDICTED
# ============================================================================
print("\n[1/6] Creating Actual vs Predicted plot...")
ax1 = plt.subplot(2, 3, 1)
plt.scatter(y, y_pred, alpha=0.3, s=10, color='#3498db')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual AQI', fontsize=12, fontweight='bold')
plt.ylabel('Predicted AQI', fontsize=12, fontweight='bold')
plt.title(f'Actual vs Predicted AQI\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text box with metrics
textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# ============================================================================
# 2. RESIDUAL PLOT
# ============================================================================
print("[2/6] Creating Residual plot...")
ax2 = plt.subplot(2, 3, 2)
residuals = y - y_pred
plt.scatter(y_pred, residuals, alpha=0.3, s=10, color='#e74c3c')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Predicted AQI', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residual Plot\n(Should be randomly scattered)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# ============================================================================
# 3. ERROR DISTRIBUTION
# ============================================================================
print("[3/6] Creating Error Distribution...")
ax3 = plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
plt.xlabel('Prediction Error', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title(f'Error Distribution\nMean Error: {residuals.mean():.2f}', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================
print("[4/6] Creating Model Comparison...")
ax4 = plt.subplot(2, 3, 4)
results_df = pd.read_csv('../models/gpu_results.csv', index_col=0)
x_pos = np.arange(len(results_df))
colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db']

bars = plt.bar(x_pos, results_df['RMSE'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Model Performance Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, results_df.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

# Add target line
plt.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target (15)', alpha=0.7)
plt.legend()

# ============================================================================
# 5. R² COMPARISON
# ============================================================================
print("[5/6] Creating R² Comparison...")
ax5 = plt.subplot(2, 3, 5)
bars = plt.bar(x_pos, results_df['R2'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Model R² Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, results_df.index, rotation=45, ha='right')
plt.ylim([0.90, 0.95])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add target line
plt.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='Target (0.95)', alpha=0.7)
plt.legend()

# ============================================================================
# 6. METRICS SUMMARY TABLE
# ============================================================================
print("[6/6] Creating Metrics Summary...")
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create table data
table_data = []
table_data.append(['Metric', 'Value', 'Target', 'Status'])
table_data.append(['RMSE', f'{rmse:.2f}', '< 15', '✗' if rmse >= 15 else '✓'])
table_data.append(['MAE', f'{mae:.2f}', '< 10', '✗' if mae >= 10 else '✓'])
table_data.append(['R²', f'{r2:.4f}', '> 0.95', '○' if r2 >= 0.94 else '✗'])
table_data.append(['', '', '', ''])
table_data.append(['Dataset', 'Value', '', ''])
table_data.append(['Records', f'{len(y):,}', '', ''])
table_data.append(['Features', f'{len(feature_cols)}', '', ''])
table_data.append(['AQI Range', f'{y.min():.0f} - {y.max():.0f}', '', ''])

table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                  colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 4):
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 0)].set_text_props(weight='bold')

# Style section header
table[(5, 0)].set_facecolor('#2ecc71')
table[(5, 0)].set_text_props(weight='bold', color='white')

plt.title('Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)

# ============================================================================
# SAVE FIGURE
# ============================================================================
plt.tight_layout()
output_path = '../models/model_performance_plots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plots to: {output_path}")

# ============================================================================
# CREATE ADDITIONAL DETAILED PLOTS
# ============================================================================
print("\n" + "="*80)
print("CREATING ADDITIONAL DETAILED PLOTS")
print("="*80)

fig2 = plt.figure(figsize=(20, 10))

# ============================================================================
# 7. AQI RANGE PERFORMANCE
# ============================================================================
print("\n[1/3] AQI Range Performance...")
ax7 = plt.subplot(1, 3, 1)

# Bin by AQI ranges
bins = [0, 50, 100, 150, 200, 300, 500]
labels = ['Good\n(0-50)', 'Moderate\n(50-100)', 'USG\n(100-150)', 
          'Unhealthy\n(150-200)', 'V.Unhealthy\n(200-300)', 'Hazardous\n(300-500)']
df['AQI_Range'] = pd.cut(y, bins=bins, labels=labels)

# Calculate RMSE for each range
range_performance = []
for label in labels:
    mask = df['AQI_Range'] == label
    if mask.sum() > 0:
        range_rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
        range_performance.append(range_rmse)
    else:
        range_performance.append(0)

colors_range = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e']
bars = plt.bar(range(len(labels)), range_performance, color=colors_range, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('AQI Category', fontsize=12, fontweight='bold')
plt.ylabel('RMSE', fontsize=12, fontweight='bold')
plt.title('Model Performance by AQI Range', fontsize=14, fontweight='bold')
plt.xticks(range(len(labels)), labels, rotation=0, fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, range_performance)):
    if val > 0:
        plt.text(bar.get_x() + bar.get_width()/2., val,
                 f'{val:.1f}',
                 ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 8. PREDICTION CONFIDENCE
# ============================================================================
print("[2/3] Prediction Confidence...")
ax8 = plt.subplot(1, 3, 2)

abs_errors = np.abs(residuals)
confidence_bins = [0, 10, 20, 30, 50, 100]
confidence_labels = ['±10', '±20', '±30', '±50', '±100']

confidence_counts = []
for i in range(len(confidence_bins)-1):
    count = ((abs_errors >= confidence_bins[i]) & (abs_errors < confidence_bins[i+1])).sum()
    confidence_counts.append(count)

percentages = [c/len(abs_errors)*100 for c in confidence_counts]
colors_conf = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']

bars = plt.bar(range(len(confidence_labels)), percentages, color=colors_conf, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('Prediction Error Range', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of Predictions (%)', fontsize=12, fontweight='bold')
plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(confidence_labels)), confidence_labels)
plt.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{pct:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# 9. IMPROVEMENT COMPARISON
# ============================================================================
print("[3/3] Improvement Comparison...")
ax9 = plt.subplot(1, 3, 3)

# Before and After comparison
categories = ['RMSE', 'MAE', 'R²']
before = [47.62, 23.01, 0.8852]  # Original results
after = [25.23, 16.08, 0.9408]   # After cleaning + engineering

x = np.arange(len(categories))
width = 0.35

bars1 = plt.bar(x - width/2, before, width, label='Before', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = plt.bar(x + width/2, after, width, label='After', color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)

plt.xlabel('Metric', fontsize=12, fontweight='bold')
plt.ylabel('Value', fontsize=12, fontweight='bold')
plt.title('Before vs After Optimization', fontsize=14, fontweight='bold')
plt.xticks(x, categories)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}' if height < 1 else f'{height:.1f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add improvement annotations
improvements = [
    ((before[0] - after[0])/before[0]*100, 0),
    ((before[1] - after[1])/before[1]*100, 1),
    ((after[2] - before[2])/before[2]*100, 2)
]

for imp, idx in improvements:
    plt.text(idx, max(before[idx], after[idx]) + 5,
             f'{"↓" if idx < 2 else "↑"}{abs(imp):.1f}%',
             ha='center', fontweight='bold', color='green' if imp > 0 or (idx == 2 and imp > 0) else 'red',
             fontsize=11)

plt.tight_layout()
output_path2 = '../models/detailed_performance_plots.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved detailed plots to: {output_path2}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\n📊 Generated plots:")
print(f"   1. {output_path}")
print(f"   2. {output_path2}")
print(f"\n✓ All visualizations saved successfully!")
