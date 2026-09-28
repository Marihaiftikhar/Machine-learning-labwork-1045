import pandas as pd

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}
df = pd.DataFrame(data)
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

zara_salary = df.loc[df["Name"] == "Zara", "Salary"].values[0]
print(f"Zara's Salary: {zara_salary}")
if zara_salary > upper_bound or zara_salary < lower_bound:
    print("Zara's salary IS an outlier by the IQR method — it is far above the rest "
          "of the salaries and would distort a mean-based summary.")
else:
    print("Zara's salary is NOT an outlier by the IQR method.")