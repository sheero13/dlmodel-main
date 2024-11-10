import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
#https://www.kaggle.com/datasets/ashydv/housing-dataset
# Load dataset
data = pd.read_csv("Housing.csv")  # Replace with your dataset path

# Data preprocessing
data = pd.get_dummies(data, drop_first=True)  # One-hot encoding for categorical features
data.fillna(data.mean(), inplace=True)  # Fill missing values

# Separate features and target variable
X = data.drop("price", axis=1)  # Replace 'SalePrice' with your target variable column
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
tree_regressor = DecisionTreeRegressor(random_state=42)
tree_regressor.fit(X_train, y_train)
y_pred_tree = tree_regressor.predict(X_test)
print("Decision Tree MAE:", mean_absolute_error(y_test, y_pred_tree))
print("Decision Tree RMSE:", mean_squared_error(y_test, y_pred_tree, squared=False))
print("Decision Tree R2:", r2_score(y_test, y_pred_tree))

# Random Forest Regressor
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
forest_regressor.fit(X_train, y_train)
y_pred_forest = forest_regressor.predict(X_test)
print("\nRandom Forest MAE:", mean_absolute_error(y_test, y_pred_forest))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_forest, squared=False))
print("Random Forest R2:", r2_score(y_test, y_pred_forest))

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
print("Best Random Forest parameters:", grid_search.best_params_)
