import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

df = pd.read_csv(url)

pakistan = df[df["Country/Region"] == "Pakistan"].copy()

pakistan["Date"] = pd.to_datetime(pakistan["Date"])

pakistan = pakistan.sort_values("Date")

pakistan["Confirmed"] = pakistan["Confirmed"].fillna(0)

plt.plot(pakistan["Date"], pakistan["Confirmed"])
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.title("COVID-19 Confirmed Cases in Pakistan")
plt.xticks(rotation=45)
plt.show()

highest_day = pakistan.loc[pakistan["Confirmed"].idxmax()]

print("Day with highest confirmed cases:")
print(highest_day[["Date", "Confirmed"]])

daily_cases = pakistan["Confirmed"].diff().fillna(pakistan["Confirmed"])

plt.boxplot(daily_cases)
plt.ylabel("Daily Cases")
plt.title("Daily COVID-19 Cases Outliers")
plt.show()

Q1 = daily_cases.quantile(0.25)
Q3 = daily_cases.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = pakistan[(daily_cases < lower) | (daily_cases > upper)]

print("Outliers in daily cases:")
print(outliers[["Date", "Confirmed"]])