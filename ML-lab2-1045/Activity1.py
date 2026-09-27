import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Medical Cost Personal Datasets.csv")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# 4. Select features and target
X = df[['age', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']
# 5. Numerical and categorical features
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['smoker', 'region']
# 6. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)
# 7. Create Linear Regression pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
# 8. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# 9. Train the Linear Regression model
model.fit(X_train, y_train)

# 10. Make predictions
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nRMSE:", rmse)
print("R² Score:", r2)
# 12. Plot Actual vs Predicted Insurance Costs
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)
# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--'
)
plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title("Actual vs Predicted Medical Insurance Costs")
plt.show()