# fit_from_csv.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import argparse
import os

parser = argparse.ArgumentParser(description="Fit a latency prediction model from CSV data and save it with a unique name.")
parser.add_argument(
    '--model_type', 
    type=str, 
    default='gradient_boosting', 
    choices=['linear', 'decision_tree', 'gradient_boosting'],
    help="Type of model to train. 'gradient_boosting' is recommended for higher accuracy."
)
parser.add_argument(
    '--scheme_name', 
    type=str, 
    default='default', 
    help="A unique name for this training scheme/experiment. This becomes part of the output filename."
)
parser.add_argument(
    '--csv_file', 
    type=str, 
    default='time_log/performance_report_new.csv',
    help="Path to the input CSV data file."
)
args = parser.parse_args()

CSV_FILENAME = args.csv_file
MODEL_FILENAME = f'dit_predictor_{args.scheme_name}_{args.model_type}.joblib'
VIS_FILENAME = f'prediction_vs_actual_{args.scheme_name}_{args.model_type}.png'

try:
    df = pd.read_csv(CSV_FILENAME)
    df.columns = df.columns.str.strip()
    print(f"Successfully loaded data from '{CSV_FILENAME}'.")
except FileNotFoundError:
    print(f"Error: Data file not found at '{CSV_FILENAME}'.")
    exit()

print(f"\nTraining scheme: '{args.scheme_name}'")
print(f"Training with model type: '{args.model_type}'")
print(f"Loaded {len(df)} data points.")

df['s'] = ((df['f'] - 1) // 4 + 1) * (df['h'] // 16) * (df['w'] // 16)
y = df['median']

if args.model_type == 'linear':
    print("Using Linear Regression model with manual feature engineering.")
    df['s^2/sp'] = (df['s']**2) / df['sp']
    df['s/sp'] = df['s'] / df['sp']
    df['s*bs/sp'] = (df['s'] * df['bs']) / df['sp']
    df['s^2*bs/sp'] = (df['s']**2 * df['bs']) / df['sp']
    df['bs^2'] = df['bs']**2
    features = ['s^2/sp', 's/sp', 'bs', 'bs^2', 's*bs/sp', 's^2*bs/sp']
    X = df[features]
    model = LinearRegression()
else:
    print("Using tree-based model. Basic features (bs, s, sp) are sufficient.")
    features = ['bs', 's', 'sp']
    X = df[features]
    if args.model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    elif args.model_type == 'gradient_boosting':
        print("Using Gradient Boosting Regressor. You can modify its hyperparameters in the script for different schemes.")
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

print("\nFitting the model...")
model.fit(X, y)
print("Model fitting complete.")

print("\n" + "="*50)
print(f"    Scheme: '{args.scheme_name}' | Model: '{args.model_type.replace('_', ' ').title()}'")
print("="*50)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"Model Fit Quality (R-squared): {r2:.6f}")
if r2 > 0.995:
    print("R² > 0.995, the model fits the data excellently!")
elif r2 < 0.9:
    print("Warning: R² is below 0.9, the model may not be accurate.")

if args.model_type == 'linear':
    print("\n### Model Coefficients ###")
    coefficients = pd.Series(model.coef_, index=features)
    print(coefficients)
    print(f"\nIntercept: {model.intercept_:.6f}")
elif hasattr(model, 'feature_importances_'):
    print("\n### Feature Importances ###")
    importances = pd.Series(model.feature_importances_, index=features)
    print(importances.sort_values(ascending=False))

joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved to '{MODEL_FILENAME}'")

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
max_val = max(y.max(), y_pred.max())
min_val = min(y.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')

plt.title(f"'{args.scheme_name}' - Predicted vs. Actual ({args.model_type.title()})", fontsize=16)
plt.xlabel('Actual Measured Latency (median)', fontsize=12)
plt.ylabel('Model Predicted Latency', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

plt.savefig(VIS_FILENAME)
print(f"Verification plot saved to '{VIS_FILENAME}'")
print("\n" + "="*50)