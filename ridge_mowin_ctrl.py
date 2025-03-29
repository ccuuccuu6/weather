import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge  # Using Ridge for regularization
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
excel_path = os.path.join(script_dir, 'campari_data.xlsx')
df = pd.read_excel(excel_path, sheet_name='datavalues')

target_col = 'Campari EMEA Sales (log difference)'

# If you have a date column, you might want to sort the DataFrame chronologically:
# df = df.sort_values('DateColumn')

# -----------------------------------------------------------------
# 2. Use Only Dummy Variables
# -----------------------------------------------------------------
dummy_cols = ['Q1', 'Q2', 'Q3']

# -----------------------------------------------------------------
# 3. Moving Window Setup
# -----------------------------------------------------------------
# Define the moving window size (using 50% of the data as window size)
window_size = int(len(df) * 0.50)

# Prepare lists to store predictions, actual values, residuals, and coefficient history
predictions = []
actuals = []
residuals = []
coefficient_history = []

# Loop over the dataset starting from the index equal to window_size
# For each iteration, the training window is the previous 'window_size' observations
for i in range(window_size, len(df)):
    # Current training window: the last 'window_size' observations before the test sample
    train_window = df.iloc[i - window_size:i].copy()
    # Current test observation
    test_sample = df.iloc[[i]]
    
    # Prepare training features (only dummy variables) and target
    X_train = train_window.drop(columns=[target_col])[dummy_cols]
    y_train = train_window[target_col]
    
    # ---------------------------
    # Fit the Ridge regression model using only dummy variables
    # ---------------------------
    model = Ridge(alpha=1.0)  # adjust alpha as needed
    model.fit(X_train, y_train)
    
    coefficient_history.append(model.coef_)
    
    # ---------------------------
    # Prepare the current test sample using only dummy variables
    # ---------------------------
    X_current = test_sample.drop(columns=[target_col])[dummy_cols]
    y_current = test_sample[target_col].values[0]
    
    # Predict the test observation
    y_pred = model.predict(X_current)[0]
    
    # Store prediction, actual value, and residual
    predictions.append(y_pred)
    actuals.append(y_current)
    residuals.append(y_current - y_pred)
    
    # Compute and print cumulative R² after this prediction
    cumulative_r2 = r2_score(actuals, predictions)
    print(f"Cumulative R² after observation {i - window_size + 1}: {cumulative_r2:.4f}")

# -----------------------------------------------------------------
# 4. Evaluation and Charts
# -----------------------------------------------------------------
# Chart 1: Predicted vs. Actual Values for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5, label='Test Predictions')
min_val = min(min(actuals), min(predictions))
max_val = max(max(actuals), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Moving Window Ridge Regression (Dummy Variables): Predicted vs. Actual')
plt.legend()
plt.show()

# Chart 2: Evolution of Coefficients Over Time
coefficient_history = np.array(coefficient_history)  # shape: (n_iterations, n_features)
plt.figure(figsize=(10, 6))
n_features = coefficient_history.shape[1]
for j in range(n_features):
    plt.plot(coefficient_history[:, j], marker='o', linestyle='-', label=f'Coef {j}')
plt.xlabel('Iteration (Test sample index)')
plt.ylabel('Coefficient value')
plt.title('Evolution of Coefficients in Moving Window Ridge Regression (Dummy Variables)')
plt.legend()
plt.show()

# Chart 3: Residual Plot for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Moving Window Ridge Regression (Dummy Variables)')
plt.legend()
plt.show()

# Overall performance on the test set
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Overall R^2 Score: {r2:.4f}')

# -----------------------------------------------------------------
# 5. Sanity Checks: Print individual results and compute manual R^2
# -----------------------------------------------------------------
print("\nSanity Check: Actual vs. Predicted and Residuals for each test observation:")
for j, (act, pred, res) in enumerate(zip(actuals, predictions, residuals)):
    print(f"Observation {j}: Actual: {act:.3f}, Predicted: {pred:.3f}, Residual: {res:.3f}")

actuals_arr = np.array(actuals)
predictions_arr = np.array(predictions)
y_bar = np.mean(actuals_arr)
sse = np.sum((actuals_arr - predictions_arr)**2)
tss = np.sum((actuals_arr - y_bar)**2)
r2_manual = 1 - (sse / tss)
print("\nManual calculation of R^2:")
print(f"R^2: {r2_manual:.4f}")

# -----------------------------------------------------------------
# 6. OLS with Statsmodels on the Final Training Window (using OLS)
# -----------------------------------------------------------------
# Use the final moving window as the training set, using only dummy variables
final_train_window = df.iloc[-window_size:].copy()
X_train = final_train_window.drop(columns=[target_col])[dummy_cols]
y_train = final_train_window[target_col]
X_train_const = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_const).fit()
print(model_sm.summary())
