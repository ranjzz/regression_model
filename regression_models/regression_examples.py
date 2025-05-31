# regression_examples.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Redirect output to file
import sys
sys.stdout = open("regression_output.txt", "w") # sys.stdout = open("regression_output.txt", "w")

# Create dataset
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Evaluation function
def evaluate_model(name, model, X_t=X_test, y_t=y_test):
    y_pred = model.predict(X_t)
    print(f"\n{name} Results:")
    print("MSE:", mean_squared_error(y_t, y_pred))
    print("R2 Score:", r2_score(y_t, y_pred))

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model("Linear Regression", lr)

# 2. Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2)
poly_reg = LinearRegression()
poly_reg.fit(X_train_p, y_train_p)
evaluate_model("Polynomial Regression", poly_reg, X_test_p, y_test_p)

# 3. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
evaluate_model("Ridge Regression", ridge)

# 4. Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
evaluate_model("Lasso Regression", lasso)

# 5. ElasticNet Regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
evaluate_model("ElasticNet Regression", elastic)

# 6. Decision Tree Regression
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
evaluate_model("Decision Tree Regression", tree)

# 7. Random Forest Regression
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train.ravel())
evaluate_model("Random Forest Regression", rf)
