import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Car Price Prediction.csv")
print(df.head())
X = df[['horsepower', 'enginesize', 'curbweight']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# 4. Create Models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Degree 2": make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    ),
    "Polynomial Degree 3": make_pipeline(
        PolynomialFeatures(degree=3),
        LinearRegression()
    ),
    "Polynomial Degree 4": make_pipeline(
        PolynomialFeatures(degree=4),
        LinearRegression()
    )
}
# 5. Train Models and Calculate Performance
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([name, rmse, r2])
# 6. Display Comparison
results = pd.DataFrame(
    results,
    columns=['Model', 'RMSE', 'R2 Score']
)
print("\nModel Comparison:")
print(results)
# 7. Fitted Curves
# Using Horsepower for visualization
X_curve = df[['horsepower']]
y_curve = df['price']
# Sort horsepower values
sort_index = X_curve['horsepower'].argsort()
X_curve = X_curve.iloc[sort_index]
y_curve = y_curve.iloc[sort_index]
plt.figure(figsize=(10, 6))
# Actual data
plt.scatter(
    X_curve,
    y_curve,
    alpha=0.5,
    label='Actual Price'
)
# Linear and Polynomial Curves
for degree in [1, 2, 3, 4]:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_curve, y_curve)

    y_curve_pred = model.predict(X_curve)

    plt.plot(
        X_curve,
        y_curve_pred,
        label=f'Degree {degree}'
    )
plt.xlabel("Horsepower")
plt.ylabel("Car Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()
plt.show()