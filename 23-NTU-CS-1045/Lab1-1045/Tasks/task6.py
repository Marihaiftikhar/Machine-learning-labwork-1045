import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

le = LabelEncoder()
df["Department_LabelEncoded"] = le.fit_transform(df["Department"])
print("Label Encoded:\n", df[["Name", "Department", "Department_LabelEncoded"]])

df_onehot = pd.get_dummies(df, columns=["Department"])
print("\nOne-Hot Encoded:\n", df_onehot)