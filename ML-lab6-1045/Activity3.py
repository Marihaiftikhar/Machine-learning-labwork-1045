
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 1. Load dataset
df = pd.read_csv("train.csv")
# 2. Explore dataset
print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:\n", df.head(10))
print("\nMissing Values:\n", df.isnull().sum().to_string())
print("\nSalePrice Statistics:\n", df["SalePrice"].describe())
# 3. Separate features and target
X = df.drop(columns=["Id", "SalePrice"])
y = df["SalePrice"]

# 4. Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns
# Preprocessing
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 5. Split dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Default Random Forest
default_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return [mae, mse, rmse, r2]

default_scores = evaluate(y_test, default_pred)

# 7. GridSearchCV
# Example grid: replace values with your assignment's table
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

optimized_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

grid = GridSearchCV(
    optimized_model,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_mse = -grid.best_score_
best_rmse = np.sqrt(best_mse)

print("Best CV MSE:", best_mse)
print("Best CV RMSE:", best_rmse)
# 8. Evaluate optimized model
best_model = grid.best_estimator_
optimized_pred = best_model.predict(X_test)
optimized_scores = evaluate(y_test, optimized_pred)
# 9. Compare models
comparison = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Default RF": default_scores,
    "Optimized RF": optimized_scores
})

print("\nModel Comparison:\n")
print(comparison.to_string(index=False))

# 10. Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, optimized_pred, alpha=0.6)

limits = [
    min(y_test.min(), optimized_pred.min()),
    max(y_test.max(), optimized_pred.max())
]
plt.plot(limits, limits, "r--")

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()

# 11. Top 5 feature importances
rf = best_model.named_steps["model"]
prep = best_model.named_steps["preprocessor"]

feature_names = prep.get_feature_names_out()

importance = pd.Series(
    rf.feature_importances_,
    index=feature_names
)

top5 = importance.sort_values(ascending=False).head(5)

print("\nTop 5 Important Features:\n", top5)

plt.figure(figsize=(8, 5))
top5.sort_values().plot(kind="barh", color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Top 5 Most Important Features")
plt.tight_layout()
plt.show()
