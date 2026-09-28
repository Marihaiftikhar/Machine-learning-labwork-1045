import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = {
    "Province": ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan"],
    "Population": [127.7, 57.9, 40.9, 14.9],
    "Literacy Rate": [64.7, 57.5, 55.1, None],
    "Region": ["East", "South", "North", "West"]
}

df = pd.DataFrame(data)

df = df.drop(index=2)

df.loc[1, "Literacy Rate"] = df["Literacy Rate"].median()

label_encoder = LabelEncoder()
df["Region_Label"] = label_encoder.fit_transform(df["Region"])

df_one_hot = pd.get_dummies(df["Region"], prefix="Region")

df = pd.concat([df, df_one_hot], axis=1)

print(df)

plt.scatter(df["Population"], df["Literacy Rate"])

for i in range(len(df)):
    plt.annotate(df["Province"].iloc[i],
                 (df["Population"].iloc[i], df["Literacy Rate"].iloc[i]))

plt.xlabel("Population (millions)")
plt.ylabel("Literacy Rate (%)")
plt.title("Population vs Literacy Rate")
plt.show()

Q1 = df["Literacy Rate"].quantile(0.25)
Q3 = df["Literacy Rate"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["Literacy Rate"] < lower) | (df["Literacy Rate"] > upper)]

print("Literacy Rate Outliers:")
print(outliers[["Province", "Literacy Rate"]])