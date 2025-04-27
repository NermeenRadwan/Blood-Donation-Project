import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from data_cleaning import clean_data

# Load and clean the data
file_path = 'transfusion.csv'
df = clean_data(file_path)

# Prepare features and target for regression
X = df[['Recency', 'Frequency', 'Time']]  # Use only non-correlated features
y = df['Monetary']  # Target variable

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge regression with GridSearchCV for alpha tuning
ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10, 100]}  # Test different regularization strengths
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best alpha from GridSearch
best_alpha = grid_search.best_params_['alpha']
print(f"Best Alpha for Ridge: {best_alpha}")

# Train Ridge model with best alpha
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)

# Predictions and evaluation for Ridge
y_pred_ridge = ridge_best.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
print("Ridge MSE: ", ridge_mse)
print("Ridge R-squared: ", ridge_r2)

# Lasso regression (L1 regularization)
lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)
print("Lasso MSE: ", lasso_mse)
print("Lasso R-squared: ", lasso_r2)

# ElasticNet regression (combines L1 and L2 regularization)
elastic_net = ElasticNet(alpha=best_alpha)
elastic_net.fit(X_train, y_train)
y_pred_en = elastic_net.predict(X_test)
en_mse = mean_squared_error(y_test, y_pred_en)
en_r2 = r2_score(y_test, y_pred_en)
print("ElasticNet MSE: ", en_mse)
print("ElasticNet R-squared: ", en_r2)

# Prepare data for plotting
models = ['Ridge', 'Lasso', 'ElasticNet']
mses = [ridge_mse, lasso_mse, en_mse]
r2_scores = [ridge_r2, lasso_r2, en_r2]

# Plot MSE and R-squared scores
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# MSE Plot
axes[0].bar(models, mses, color=['blue', 'orange', 'green'])
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_ylabel('MSE')
axes[0].set_xlabel('Models')

# R-squared Plot
axes[1].bar(models, r2_scores, color=['blue', 'orange', 'green'])
axes[1].set_title('R-squared Scores')
axes[1].set_ylabel('R-squared')
axes[1].set_xlabel('Models')

# Show plots
plt.tight_layout()
plt.show()
