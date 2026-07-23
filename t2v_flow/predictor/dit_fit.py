# fit_final_data.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib
import json
matplotlib.use('Agg')
# ==============================================================================
# ==============================================================================
data = [
    (46800, 1, 111.47),
    (25200, 2, 81.67),
    (25200, 3, 123.21),
    (61200, 2, 350.48),
    (61200, 1, 173.26),
    (36000, 2, 145.91),
    (118800, 1, 592.00),
    (82800, 1, 292.74),
    (7200, 1, 8.037),
    (54000, 1, 139.74),
    (54000, 2, 279.81),
    (118800, 2, 1150.02),
    (46800, 2, 221.66),
    (82800, 2, 582.37),
    (18000, 1, 24.75),
    (7200, 2, 15.48),
    (7200, 3, 22.65),
    (25200, 1, 42.00),
    (18000, 3, 74.10),
]

df = pd.DataFrame(data, columns=['s', 'bs', 'T_layer'])

print("Data successfully transcribed based on your clarification.")
print(f"Loaded {len(df)} data points.")
print("Data preview:")
print(df.head())


# ==============================================================================
# T_layer ≈ c1*s² + c2*s + c3*bs + c4*bs² + c5*(s*bs) + c6*(s²*bs) + c7
# ==============================================================================
df['s^2'] = df['s']**2
df['bs^2'] = df['bs']**2
df['s*bs'] = df['s'] * df['bs']
df['s^2*bs'] = df['s']**2 * df['bs']

features = ['s^2', 's', 'bs', 'bs^2', 's*bs', 's^2*bs']
X = df[features]
y = df['T_layer']


# ==============================================================================
# ==============================================================================
model = LinearRegression()
model.fit(X, y)


# ==============================================================================
# ==============================================================================
print("\n" + "="*50)
print("          Model Fitting Results")
print("="*50)

coefficients = pd.Series(model.coef_, index=features)
intercept = model.intercept_

print("\n### Model Coefficients ###\n")
print(f"c1 (for s^2):    {coefficients['s^2']:.6e}")
print(f"c2 (for s):      {coefficients['s']:.6e}")
print(f"c3 (for bs):     {coefficients['bs']:.6e}")
print(f"c4 (for bs^2):   {coefficients['bs^2']:.6e}")
print(f"c5 (for s*bs):   {coefficients['s*bs']:.6e}")
print(f"c6 (for s^2*bs): {coefficients['s^2*bs']:.6e}")
print(f"c7 (intercept):  {intercept:.6f}")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nModel Fit Quality (R-squared): {r2:.6f}")
if r2 > 0.99:
    print("R² > 0.99, the model fits the data excellently!")
elif r2 > 0.95:
    print("R² > 0.95, the model provides a good fit.")
else:
    print("R² is lower than expected. Please double-check the data for any outliers.")

model_weights = {
    "coefficients": coefficients.to_dict(),
    "intercept": intercept
}
weights_filename = 'dit_cost_model_weights_final.json'
with open(weights_filename, 'w') as f:
    json.dump(model_weights, f, indent=4)
print(f"\nModel weights have been successfully saved to '{weights_filename}'")


# ==============================================================================
# ==============================================================================
plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.7, edgecolors='k', label='Data Points')
max_val = max(y.max(), y_pred.max())
min_val = min(y.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit (y=x)')

plt.title('Predicted vs. Actual Values', fontsize=16)
plt.xlabel('Actual Measured Time (T_layer)', fontsize=12)
plt.ylabel('Model Predicted Time', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

vis_filename = 'prediction_vs_actual_final.png'
plt.savefig(vis_filename)
print(f"Verification plot has been saved to '{vis_filename}'")
print("\n" + "="*50)