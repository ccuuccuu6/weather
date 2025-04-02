import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge  # Using Ridge for regularization
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------
# 1. Load Data and Separate Actual vs. Future Data
# -----------------------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
excel_path = os.path.join(script_dir, 'campari_data.xlsx')
df = pd.read_excel(excel_path, sheet_name='datavalues')

target_col = 'Campari EMEA Sales (log difference)'

# Separate rows with actual Sales values from those with missing Sales (future data)
df_actual = df[df[target_col].notna()].copy()
df_future = df[df[target_col].isna()].copy()

# -----------------------------------------------------------------
# 2. Define Columns and Setup for Expanding Window on Actual Data
# -----------------------------------------------------------------
continuous_cols = ['Campari EMEA Precipitation (log)', 
                   'Campari EMEA Mean Temp (log)']
dummy_cols = ['Q1', 'Q2', 'Q3']
feature_names = continuous_cols + dummy_cols

# Use 20% of the actual data as the initial training set (ordered by index)
train_size = int(len(df_actual) * 0.75)
df_train = df_actual.iloc[:train_size].copy()  # initial training set
df_test = df_actual.iloc[train_size:].copy()   # test set (to be forecasted sequentially)

# Prepare lists to store predictions, actual values, residuals, etc.
predictions = []
actuals = []
residuals = []
coefficient_history = []
feature_contributions_history = []
prediction_intervals_lower = []
prediction_intervals_upper = []
test_indices = df_test.index.tolist()  # preserve original test indices for plotting

# -----------------------------------------------------------------
# 3. Expanding Window Loop using Ridge Regression (Actual Data)
# -----------------------------------------------------------------
for i, idx in enumerate(test_indices, start=1):
    # Prepare training features and target from current training set
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    # Preprocessing for training data
    scaler = StandardScaler()
    X_train_cont = X_train[continuous_cols]
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    
    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_train_poly_cont = poly.fit_transform(X_train_cont_scaled)
    
    X_train_dummy = X_train[dummy_cols].values
    X_train_final = np.concatenate([X_train_poly_cont, X_train_dummy], axis=1)
    
    # Fit the Ridge regression model
    model = Ridge(alpha=1.0)  # adjust alpha as needed
    model.fit(X_train_final, y_train)
    coefficient_history.append(model.coef_)
    
    # Prepare and transform the current test sample
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
    
    # Bootstrapping for Prediction Interval (90%)
    n_bootstraps = 1000
    boot_predictions = []
    model_residuals = y_train - model.predict(X_train_final)
    for _ in range(n_bootstraps):
        sampled_resid = np.random.choice(model_residuals)
        boot_predictions.append(y_pred + sampled_resid)
    boot_predictions = np.array(boot_predictions)
    lower_bound = np.percentile(boot_predictions, 5)
    upper_bound = np.percentile(boot_predictions, 95)
    
    prediction_intervals_lower.append(lower_bound)
    prediction_intervals_upper.append(upper_bound)
    
    print(f"Observation {i}: Prediction: {y_pred:.4f}, 90% PI: [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    # Calculate feature contributions (each feature's value * its coefficient, plus intercept)
    contributions = (X_current_final.flatten() * model.coef_)
    intercept_contribution = model.intercept_
    contributions_with_intercept = np.concatenate(([intercept_contribution], contributions))
    feature_contributions_history.append(contributions_with_intercept)
    
    # Store prediction, actual value, and residual
    predictions.append(y_pred)
    actuals.append(y_current)
    residuals.append(y_current - y_pred)
    
    cumulative_r2 = r2_score(actuals, predictions)
    print(f"Cumulative R² after observation {i}: {cumulative_r2:.4f}")
    
    # Expand the training set by appending the current test observation
    df_train = pd.concat([df_train, current_test])

# -----------------------------------------------------------------
# 4. Forecast Future Data (Rows with Missing Sales)
# -----------------------------------------------------------------
# Fit final model using all actual data (df_train)
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
scaler_final = StandardScaler()
X_train_cont = X_train[continuous_cols]
X_train_cont_scaled = scaler_final.fit_transform(X_train_cont)
poly_final = PolynomialFeatures(degree=1, include_bias=False)
X_train_poly_cont = poly_final.fit_transform(X_train_cont_scaled)
X_train_dummy = X_train[dummy_cols].values
X_train_final = np.concatenate([X_train_poly_cont, X_train_dummy], axis=1)

model_final = Ridge(alpha=1.0)
model_final.fit(X_train_final, y_train)
final_residuals = y_train - model_final.predict(X_train_final)

future_predictions = []
future_PI_lower = []
future_PI_upper = []
future_PI_lower_75 = []  # For 75% PI
future_PI_upper_75 = []  # For 75% PI

for idx, row in df_future.iterrows():
    X_row = row.drop(labels=[target_col])
    X_row_cont = np.array(X_row[continuous_cols]).reshape(1, -1)
    X_row_cont_scaled = scaler_final.transform(X_row_cont)
    X_row_poly_cont = poly_final.transform(X_row_cont_scaled)
    X_row_dummy = np.array(X_row[dummy_cols]).reshape(1, -1)
    X_row_final = np.concatenate([X_row_poly_cont, X_row_dummy], axis=1)
    
    y_pred_future = model_final.predict(X_row_final)[0]
    
    boot_preds = []
    for _ in range(n_bootstraps):
        sampled_resid = np.random.choice(final_residuals)
        boot_preds.append(y_pred_future + sampled_resid)
    boot_preds = np.array(boot_preds)
    
    # 90% prediction interval
    lower_bound_future = np.percentile(boot_preds, 5)
    upper_bound_future = np.percentile(boot_preds, 95)
    # 75% prediction interval
    lower_bound_future_75 = np.percentile(boot_preds, 12.5)
    upper_bound_future_75 = np.percentile(boot_preds, 87.5)
    
    future_predictions.append(y_pred_future)
    future_PI_lower.append(lower_bound_future)
    future_PI_upper.append(upper_bound_future)
    future_PI_lower_75.append(lower_bound_future_75)
    future_PI_upper_75.append(upper_bound_future_75)
    
    print(f"Future Observation {idx}: Prediction: {y_pred_future:.4f}, 90% PI: [{lower_bound_future:.4f}, {upper_bound_future:.4f}], 75% PI: [{lower_bound_future_75:.4f}, {upper_bound_future_75:.4f}]")

# -----------------------------------------------------------------
# 5. New Line Graph: Actual and Predicted Values with 90% and 75% PI for Future Data
# -----------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Plot actual Sales only for the test set (where predictions exist)
plt.plot(df_test.index, df_test[target_col], 'o-', label='Actual Sales (Test)')
# Plot test set predictions (from expanding window)
plt.plot(df_test.index, predictions, 'o-', label='Predicted Sales (Test)')

# Connect last test observation with future forecasts
last_actual_index = df_test.index[-1]
last_actual_value = df_test[target_col].iloc[-1]
# Combine last test point with forecasted indices and predictions
combined_indices = np.concatenate(([last_actual_index], df_future.index))
combined_forecast = np.concatenate(([last_actual_value], future_predictions))
plt.plot(combined_indices, combined_forecast, 'o-', label='Predicted Sales (Future)')

# For shading, extend the PI boundaries to start from the last test actual value.
combined_lower_90 = np.concatenate(([last_actual_value], np.array(future_PI_lower)))
combined_upper_90 = np.concatenate(([last_actual_value], np.array(future_PI_upper)))
combined_lower_75 = np.concatenate(([last_actual_value], np.array(future_PI_lower_75)))
combined_upper_75 = np.concatenate(([last_actual_value], np.array(future_PI_upper_75)))

# Shade the 90% prediction interval
plt.fill_between(combined_indices, combined_lower_90, combined_upper_90, 
                 color='blue', alpha=0.2, label='90% PI (Future)')
# Shade the 75% prediction interval
plt.fill_between(combined_indices, combined_lower_75, combined_upper_75, 
                 color='green', alpha=0.2, label='75% PI (Future)')

plt.xlabel('Index')
plt.ylabel('Campari EMEA Sales (log difference)')
plt.title('Actual and Predicted Sales with 90% & 75% Prediction Intervals')
plt.legend()
plt.show()

# -----------------------------------------------------------------
# 6. Additional Charts from Previous Code
# -----------------------------------------------------------------
# Chart A: Predicted vs. Actual Values for Test Data
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

# Chart B: Evolution of Coefficients Over Time
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

# Chart C: Residual Plot for Test Data
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Expanding Window Ridge Regression')
plt.legend()
plt.show()

# Chart D: Stacked Bar Chart of Feature Contributions
all_feature_names = ['Intercept'] + feature_names
feature_contributions_history = np.array(feature_contributions_history)
period_indices = np.arange(len(feature_contributions_history))
plt.figure(figsize=(12, 8))
pos_bottom = np.zeros(len(feature_contributions_history))
neg_bottom = np.zeros(len(feature_contributions_history))
for j, fname in enumerate(all_feature_names):
    seg = feature_contributions_history[:, j]
    pos_seg = np.where(seg > 0, seg, 0)
    neg_seg = np.where(seg < 0, seg, 0)
    bar_handle = plt.bar(period_indices, pos_seg, bottom=pos_bottom, label=fname if np.any(pos_seg) else None)
    pos_bottom += pos_seg
    plt.bar(period_indices, neg_seg, bottom=neg_bottom, color=bar_handle[0].get_facecolor(), label=fname if np.any(neg_seg) and np.all(pos_seg==0) else None)
    neg_bottom += neg_seg

plt.plot(period_indices, predictions, marker='D', linestyle='None', markersize=8, color='black', label='Predicted Value')
plt.plot(period_indices, actuals, marker='X', linestyle='None', markersize=8, color='red', label='Actual Value')
plt.xlabel('Test Period Index')
plt.ylabel('Contribution to Prediction')
plt.title('Stacked Bar Chart of Feature Contributions (with Intercept) by Test Period')
plt.legend()
plt.show()

# Chart E: Predicted Values with 90% Prediction Intervals for Test Data
plt.figure(figsize=(10, 6))
lower_errors = np.array(predictions) - np.array(prediction_intervals_lower)
upper_errors = np.array(prediction_intervals_upper) - np.array(predictions)
yerr = [lower_errors, upper_errors]
plt.errorbar(range(len(predictions)), predictions, yerr=yerr, fmt='o', color='blue', ecolor='gray', capsize=3, label='Prediction with 90% PI')
plt.plot(range(len(predictions)), predictions, 'b-', label='Predicted Value')
plt.xlabel('Test Period Index')
plt.ylabel('Predicted Value')
plt.title('Predicted Values with 90% Prediction Intervals (Test Data)')
plt.legend()
plt.show()

# -----------------------------------------------------------------
# 7. Evaluation and OLS with Statsmodels on the Final Training Set
# -----------------------------------------------------------------
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
print(f'Mean Squared Error (Actual Data): {mse:.4f}')
print(f'Overall R^2 Score (Actual Data): {r2:.4f}')

print("\nSanity Check: Actual vs. Predicted and Residuals for each test observation:")
for j, (act, pred, res) in enumerate(zip(actuals, predictions, residuals)):
    print(f"Observation {j}: Actual: {act:.3f}, Predicted: {pred:.3f}, Residual: {res:.3f}")

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
