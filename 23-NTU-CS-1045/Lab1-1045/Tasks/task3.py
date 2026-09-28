import json
import pandas as pd
import matplotlib.pyplot as plt

students = pd.read_csv("students.csv")
with open("attendance.json") as f:
    attendance_json = json.load(f)
attendance = pd.DataFrame(list(attendance_json.items()), columns=["Name", "Attendance"])
extra = pd.read_excel("extra.xlsx")
merged = students.merge(attendance, on="Name").merge(extra, on="Name")

colors = ["red" if a < 70 else "blue" for a in merged["Attendance"]]
plt.figure(figsize=(8, 6))
plt.scatter(merged["Marks"], merged["Attendance"], c=colors, s=100)
for _, row in merged.iterrows():
    plt.annotate(row["Name"], (row["Marks"], row["Attendance"]))
plt.axhline(70, color="gray", linestyle="--", label="70% Attendance Threshold")
plt.xlabel("Marks")
plt.ylabel("Attendance (%)")
plt.title("Marks vs Attendance (Red = Attendance < 70%)")
plt.legend()
plt.tight_layout()
plt.show()