
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

data = {
    "ID": range(1, 101),
    "Math": np.random.randint(40, 101, 100),
    "Science": np.random.randint(40, 101, 100),
    "English": np.random.randint(40, 101, 100)
}

df = pd.DataFrame(data)

df["Grade"] = np.random.choice(["A", "B", "C"], 100)

df.loc[np.random.choice(df.index, 5, replace=False), "Math"] = np.nan
df.loc[np.random.choice(df.index, 5, replace=False), "Science"] = np.nan
df.loc[np.random.choice(df.index, 5, replace=False), "English"] = np.nan

df["Math"] = df["Math"].fillna(df["Math"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

df["Grade_Encoded"] = df["Grade"].map({"C": 0, "B": 1, "A": 2})

plt.hist(df["Math"], bins=10)
plt.xlabel("Math Scores")
plt.ylabel("Number of Students")
plt.title("Math Scores")
plt.show()

df["Total"] = df["Math"] + df["Science"] + df["English"]

plt.boxplot(df["Total"])
plt.ylabel("Total Score")
plt.title("Total Score Boxplot")
plt.show()

Q1 = df["Total"].quantile(0.25)
Q3 = df["Total"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Total"] < lower) | (df["Total"] > upper)]

print("Unusually high/low total scores:")
print(outliers[["ID", "Total"]])

print("\nAverage Total Score by Grade:")
print(df.groupby("Grade")["Total"].mean())

print("\nGrade with Best Overall Performance:")
print(df.groupby("Grade")["Total"].mean().idxmax())