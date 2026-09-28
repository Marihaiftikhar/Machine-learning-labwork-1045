import json
import pandas as pd

students = pd.read_csv("students.csv")
with open("attendance.json") as f:
    attendance_json = json.load(f)
attendance = pd.DataFrame(list(attendance_json.items()), columns=["Name", "Attendance"])
extra = pd.read_excel("extra.xlsx")

merged = students.merge(attendance, on="Name").merge(extra, on="Name")
merged = merged[["Name", "Marks", "Attendance", "Bonus"]]
print("Merged DataFrame:\n", merged)