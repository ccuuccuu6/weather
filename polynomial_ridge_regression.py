import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
excel_path = os.path.join(script_dir, 'campari_data.xlsx')

df = pd.read_excel(excel_path, sheet_name='datavalues')

target_col = 'Campari EMEA Sales (log difference)'
X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------------------------------------------
# 2. Identify Continuous vs. Dummy Columns
# -----------------------------------------------------------------
# Change these lists to match your actual column names
continuous_cols = ['Campari EMEA Precipitation (log)', 
                   'Campari EMEA Mean Temp (log)']
dummy_cols = ['Q1', 'Q2', 'Q3']

# -----------------------------------------------------------------
# 3. Train-Test Split
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------------------------------------------
# 4. Separate and Scale Continuous Columns Only
# -----------------------------------------------------------------
X_train_cont = X_train[continuous_cols]
X_test_cont = X_test[continuous_cols]

scaler = StandardScaler()
X_train_cont_scaled = scaler.fit_transform(X_train_cont)
X_test_cont_scaled = scaler.transform(X_test_cont)

# -----------------------------------------------------------------
# 5. Create Polynomial Features for Continuous Columns Only
# -----------------------------------------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_cont = poly.fit_transform(X_train_cont_scaled)
X_test_poly_cont = poly.transform(X_test_cont_scaled)

# -----------------------------------------------------------------
# 6. Concatenate Dummy Variables Back
# -----------------------------------------------------------------
X_train_dummy = X_train[dummy_cols].values  # as NumPy array
X_test_dummy = X_test[dummy_cols].values

# Combine the polynomial-transformed continuous features with the dummy columns
X_train_final = np.concatenate([X_train_poly_cont, X_train_dummy], axis=1)
X_test_final = np.concatenate([X_test_poly_cont, X_test_dummy], axis=1)

# -----------------------------------------------------------------
# 7. Fit the Polynomial Regression Model
# -----------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_final, y_train)

# -----------------------------------------------------------------
# 8. Charts and Evaluation
# -----------------------------------------------------------------
# Chart 1: Predicted vs. Actual
plt.figure(figsize=(10, 6))
y_pred_train = model.predict(X_train_final)
y_pred_test = model.predict(X_test_final)

plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5, label='Test data')

min_val = min(y.min(), y_pred_train.min(), y_pred_test.min())
max_val = max(y.max(), y_pred_train.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal fit')

plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Polynomial Regression (No Dummy Expansion): Predicted vs. Actual')
plt.legend()
plt.show()

# Chart 2: Coefficient Plot
plt.figure(figsize=(10, 6))
plt.plot(model.coef_, marker='o', linestyle='none', label='Coefficients')
plt.xlabel('Feature index')
plt.ylabel('Coefficient value')
plt.title('Polynomial Regression Coefficients (No Dummy Expansion)')
plt.legend()
plt.show()

# Chart 3: Residual Plot
residuals = y_test - model.predict(X_test_final)
plt.figure(figsize=(10, 6))
plt.scatter(model.predict(X_test_final), residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero error')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
plt.show()

# Evaluate model performance on the test data
y_pred = model.predict(X_test_final)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# -----------------------------------------------------------------
# 9. OLS with Statsmodels
# -----------------------------------------------------------------
# Add a constant for the intercept
X_train_final_const = sm.add_constant(X_train_final)
model_sm = sm.OLS(y_train, X_train_final_const).fit()
print(model_sm.summary())
