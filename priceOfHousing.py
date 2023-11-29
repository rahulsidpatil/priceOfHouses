import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Preprocessing: Handling missing values and dropping columns with a high percentage of missing values
columns_to_drop = data.columns[data.isnull().mean() > 0.5]
data_cleaned = data.drop(columns=columns_to_drop)

# Fill remaining missing values
for column in data_cleaned.columns:
    if data_cleaned[column].dtype == 'object':
        data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)
    else:
        data_cleaned[column].fillna(data_cleaned[column].median(), inplace=True)

# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data_cleaned.drop('Id', axis=1))

# Feature Selection: Using correlation with SalePrice
corr_matrix = data_encoded.corr()
saleprice_corr = corr_matrix['SalePrice'].sort_values(ascending=False)

# Splitting the dataset into training and test sets
X = data_encoded.drop('SalePrice', axis=1)
y = data_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning: Finding optimal alpha for Ridge and Lasso Regression
parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 50]}
ridge_cv = GridSearchCV(Ridge(), parameters, scoring='neg_mean_squared_error', cv=5)
lasso_cv = GridSearchCV(Lasso(max_iter=10000), parameters, scoring='neg_mean_squared_error', cv=5)

ridge_cv.fit(X_train_scaled, y_train)
lasso_cv.fit(X_train_scaled, y_train)

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
# ridge_cv = GridSearchCV(Ridge(), parameters, scoring='neg_mean_squared_error', cv=5)
# lasso_cv = GridSearchCV(Lasso(), parameters, scoring='neg_mean_squared_error', cv=5)

# ridge_cv.fit(X_train_scaled, y_train)
# lasso_cv.fit(X_train_scaled, y_train)

# Model Evaluation: Using best alpha values
ridge_best = Ridge(alpha=ridge_cv.best_params_['alpha'])
lasso_best = Lasso(alpha=lasso_cv.best_params_['alpha'])

ridge_best.fit(X_train_scaled, y_train)
lasso_best.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_best.predict(X_test_scaled)
y_pred_lasso = lasso_best.predict(X_test_scaled)

# Metrics: RMSE and R-squared
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

# Identifying Significant Features
ridge_coeffs = pd.DataFrame({'Feature': X.columns, 'Ridge Coefficient': ridge_best.coef_})
lasso_coeffs = pd.DataFrame({'Feature': X.columns, 'Lasso Coefficient': lasso_best.coef_})

coeffs_combined = pd.merge(ridge_coeffs, lasso_coeffs, on='Feature')
coeffs_combined['Ridge Coefficient Absolute'] = coeffs_combined['Ridge Coefficient'].abs()
significant_features = coeffs_combined.sort_values('Ridge Coefficient Absolute', ascending=False).head(10)

print("Ridge RMSE:", rmse_ridge)
print("Ridge R2:", r2_ridge)
print("Lasso RMSE:", rmse_lasso)
print("Lasso R2:", r2_lasso)
print("Significant Features:\n", significant_features)
