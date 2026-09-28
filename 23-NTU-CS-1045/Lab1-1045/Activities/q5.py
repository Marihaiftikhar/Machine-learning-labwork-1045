import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
print("\nAfter Handling Missing Values:\n", df)
le = LabelEncoder()
df["Department_LabelEncoded"] = le.fit_transform(df["Department"])
print("\nLabel Encoded Department:\n", df[["Name", "Department", "Department_LabelEncoded"]])
 
df_onehot = pd.get_dummies(df, columns=["Department"])
print("\nOne-Hot Encoded Department:\n", df_onehot)
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
print(f"\nIQR bounds -> lower: {lower_bound}, upper: {upper_bound}")
print("Salary outliers (IQR method):\n", outliers)
zara_salary = df.loc[df["Name"] == "Zara", "Salary"].values[0]
print(f"\nZara's Salary: {zara_salary}")
if zara_salary > upper_bound or zara_salary < lower_bound:
    print("Zara's salary IS an outlier by the IQR method — it is far above the rest "
          "of the salaries in the dataset and would distort a mean-based summary.")
else:
    print("Zara's salary is NOT flagged as an outlier by the IQR method.")
