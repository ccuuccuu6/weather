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
# 2. Identify Dummy Columns
# -----------------------------------------------------------------
dummy_cols = ['Q1', 'Q2', 'Q3']

# -----------------------------------------------------------------
# 3. Expanding Window Setup
# -----------------------------------------------------------------
# Use 20% of the data as the initial training set
train_size = int(len(df) * 0.2)
df_train = df.iloc[:train_size].copy()  # initial training set
df_test = df.iloc[train_size:].copy()   # test set (to be forecasted sequentially)

# Prepare lists to store test predictions, actual values, residuals, coefficient history,
# and feature contributions history
predictions = []
actuals = []
residuals = []
coefficient_history = []
feature_contributions_history = []
test_indices = df_test.index.tolist()  # preserve the original test indices for plotting

# -----------------------------------------------------------------
# 4. Expanding Window Loop using Ridge Regression
# -----------------------------------------------------------------
# Enumerate to get a counter for printing the cumulative R² after each forecast.
for i, idx in enumerate(test_indices, start=1):
    # Prepare training features and target from current training set
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    # Use only dummy variables for training
    X_train_dummy = X_train[dummy_cols].values
    X_train_final = X_train_dummy
    
    # ---------------------------
    # Fit the Ridge regression model
    # ---------------------------
    model = Ridge(alpha=1.0)  # adjust alpha as needed
    model.fit(X_train_final, y_train)
    
    coefficient_history.append(model.coef_)
    
    # ---------------------------
    # Prepare and transform the current test sample
    # ---------------------------
    current_test = df_test.loc[[idx]]
    X_current = current_test.drop(columns=[target_col])
    y_current = current_test[target_col].values[0]
    
    X_current_dummy = X_current[dummy_cols].values
    X_current_final = X_current_dummy
    
    # Predict the test observation
    y_pred = model.predict(X_current_final)[0]
    
    # Calculate feature contributions: each feature's value times its coefficient
    contributions = (X_current_final.flatten() * model.coef_)
    intercept_contribution = model.intercept_
    contributions_with_intercept = np.concatenate(([intercept_contribution], contributions))
    feature_contributions_history.append(contributions_with_intercept)
    
    # Store prediction, actual value, and residual
    predictions.append(y_pred)
    actuals.append(y_current)
    residuals.append(y_current - y_pred)
    
    # Compute and print cumulative R² after this prediction
    cumulative_r2 = r2_score(actuals, predictions)
    print(f"Cumulative R² after observation {i}: {cumulative_r2:.4f}")
    
    # Expand the training set by appending the current test observation
    df_train = pd.concat([df_train, current_test])

# -----------------------------------------------------------------
# 5. Evaluation and Charts
# -----------------------------------------------------------------
# Chart 1: Predicted vs. Actual Values for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5, label='Test Predictions')
min_val = min(min(actuals), min(predictions))
max_val = max(max(actuals), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal fit')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Expanding Window Ridge Regression: Predicted vs. Actual')
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
plt.title('Evolution of Coefficients in Expanding Window Ridge Regression')
plt.legend()
plt.show()

# Chart 3: Residual Plot for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Expanding Window Ridge Regression')
plt.legend()
plt.show()

# Chart 4: Absolute contribution of features
# Define the feature names: intercept plus dummy variables
all_feature_names = ['Intercept'] + dummy_cols

# Convert contributions history to an array: shape (n_periods, n_features+1)
feature_contributions_history = np.array(feature_contributions_history)
period_indices = np.arange(len(feature_contributions_history))

plt.figure(figsize=(12, 8))
# Initialize accumulators for positive and negative contributions separately.
pos_bottom = np.zeros(len(feature_contributions_history))
neg_bottom = np.zeros(len(feature_contributions_history))

# Loop over each feature and plot positive and negative parts separately.
for j, fname in enumerate(all_feature_names):
    seg = feature_contributions_history[:, j]
    pos_seg = np.where(seg > 0, seg, 0)
    neg_seg = np.where(seg < 0, seg, 0)
    
    # Plot positive contributions starting from 0 upward.
    bar_handle = plt.bar(period_indices, pos_seg, bottom=pos_bottom, label=fname if np.any(pos_seg) else None)
    pos_bottom += pos_seg
    
    # Plot negative contributions starting from 0 downward, using the same color.
    plt.bar(period_indices, neg_seg, bottom=neg_bottom, color=bar_handle[0].get_facecolor(), label=fname if np.any(neg_seg) and np.all(pos_seg==0) else None)
    neg_bottom += neg_seg

# Add markers for predicted values (diamond, black) and actual values (cross, red).
plt.plot(period_indices, predictions, marker='D', linestyle='None', markersize=8, color='black', label='Predicted Value')
plt.plot(period_indices, actuals, marker='X', linestyle='None', markersize=8, color='red', label='Actual Value')

plt.xlabel('Test Period Index')
plt.ylabel('Contribution to Prediction')
plt.title('Stacked Bar Chart of Feature Contributions (with Intercept) by Test Period')
plt.legend()
plt.show()

# Overall performance on the test set
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Overall R^2 Score: {r2:.4f}')

# -----------------------------------------------------------------
# 5A. Sanity Checks: Print individual results and compute manual R^2
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
# 6. OLS with Statsmodels on the Final Training Set (using OLS)
# -----------------------------------------------------------------
# Use only dummy variables for the final model
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
X_train_dummy = X_train[dummy_cols].values
X_train_final = X_train_dummy
X_train_final_const = sm.add_constant(X_train_final)
model_sm = sm.OLS(y_train, X_train_final_const).fit()
print(model_sm.summary())
