import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Position_Salaries.csv")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumns:")
print(df.columns)
X = df[['Level']]
y = df['Salary']
# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# 4. Create Models
models = {
    
    "Linear Regression":
        LinearRegression(),

    "Polynomial Degree 2":
        make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        ),

    "Polynomial Degree 3":
        make_pipeline(
            PolynomialFeatures(degree=3),
            LinearRegression()
        ),

    "Polynomial Degree 4":
        make_pipeline(
            PolynomialFeatures(degree=4),
            LinearRegression()
        )
}
# 5. Train Models and Calculate Performance
results = []

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    # Calculate R2 Score
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append([name, rmse, r2])
# 6. Display Model Comparison
results = pd.DataFrame(
    results,
    columns=['Model', 'RMSE', 'R2 Score']
)

print("\nModel Comparison:")
print(results)
# 7. Prepare Data for Fitted Curves
X_curve = df[['Level']]
y_curve = df['Salary']
# Sort Level values
sort_index = X_curve['Level'].argsort()

X_curve = X_curve.iloc[sort_index]
y_curve = y_curve.iloc[sort_index]


# 8. Plot Actual Data
plt.figure(figsize=(10, 6))

plt.scatter(
    X_curve,
    y_curve,
    label='Actual Salary'
)
# 9. Plot Linear and Polynomial Regression Curves
for degree in [1, 2, 3, 4]:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    # Train model
    model.fit(X_curve, y_curve)

    # Predictions
    y_curve_pred = model.predict(X_curve)

    # Plot fitted curve
    plt.plot(
        X_curve,
        y_curve_pred,
        label=f'Degree {degree}'
    )
# 10. Graph Labels
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid()
plt.show()