import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

plt.figure(figsize=(6, 6))
plt.boxplot(df["Salary"], vert=True)
plt.title("Boxplot of Salary")
plt.ylabel("Salary")
plt.show()

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]
print(f"IQR bounds -> lower: {lower_bound}, upper: {upper_bound}")
print("Salary outliers:\n", outliers)