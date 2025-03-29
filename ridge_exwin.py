import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
# 2. Identify Continuous vs. Dummy Columns
# -----------------------------------------------------------------
continuous_cols = ['Campari EMEA Precipitation (log)', 
                   'Campari EMEA Mean Temp (log)']
dummy_cols = ['Q1', 'Q2', 'Q3']

# -----------------------------------------------------------------
# 3. Expanding Window Setup
# -----------------------------------------------------------------
# Use 20% of the data as the initial training set (ordered by index)
train_size = int(len(df) * 0.2)
df_train = df.iloc[:train_size].copy()  # initial training set
df_test = df.iloc[train_size:].copy()   # test set (to be forecasted sequentially)

# Prepare lists to store test predictions, actual values, residuals, and coefficient history
predictions = []
actuals = []
residuals = []
coefficient_history = []
test_indices = df_test.index.tolist()  # preserve the original test indices for plotting

# -----------------------------------------------------------------
# 4. Expanding Window Loop using Ridge Regression
# -----------------------------------------------------------------
# Enumerate to get a counter for printing the cumulative R² after each forecast.
for i, idx in enumerate(test_indices, start=1):
    # Prepare training features and target from current training set
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    # ---------------------------
    # Preprocessing for training data
    # ---------------------------
    scaler = StandardScaler()
    X_train_cont = X_train[continuous_cols]
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_train_poly_cont = poly.fit_transform(X_train_cont_scaled)
    
    X_train_dummy = X_train[dummy_cols].values
    X_train_final = np.concatenate([X_train_poly_cont, X_train_dummy], axis=1)
    
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
    
    X_current_cont = X_current[continuous_cols]
    X_current_cont_scaled = scaler.transform(X_current_cont)
    X_current_poly_cont = poly.transform(X_current_cont_scaled)
    X_current_dummy = X_current[dummy_cols].values
    X_current_final = np.concatenate([X_current_poly_cont, X_current_dummy], axis=1)
    
    # Predict the test observation
    y_pred = model.predict(X_current_final)[0]
    
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
# (This uses the final expanded training set; note that statsmodels doesn't offer Ridge with summary output)
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
scaler = StandardScaler()
X_train_cont = X_train[continuous_cols]
X_train_cont_scaled = scaler.fit_transform(X_train_cont)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_cont = poly.fit_transform(X_train_cont_scaled)
X_train_dummy = X_train[dummy_cols].values
X_train_final = np.concatenate([X_train_poly_cont, X_train_dummy], axis=1)
X_train_final_const = sm.add_constant(X_train_final)
model_sm = sm.OLS(y_train, X_train_final_const).fit()
print(model_sm.summary())
