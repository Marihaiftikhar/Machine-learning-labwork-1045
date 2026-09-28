import json
import pandas as pd
import matplotlib.pyplot as plt
students_df = pd.DataFrame({
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor", "Usman", "Ayesha"],
    "Marks": [78, 85, 62, 90, 55, 73, 40, 88]
})
students_df.to_csv("students.csv", index=False)

attendance_data = {
    "Ahsan": 92, "Hira": 65, "Bilal": 80, "Zara": 55,
    "Salman": 60, "Mahnoor": 95, "Usman": 45, "Ayesha": 88
}
with open("attendance.json", "w") as f:
    json.dump(attendance_data, f)

extra_df = pd.DataFrame({
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor", "Usman", "Ayesha"],
    "Bonus": [5, 3, 0, 2, 4, 5, 1, 3]
})
extra_df.to_excel("extra.xlsx", index=False)
students = pd.read_csv("students.csv")

with open("attendance.json") as f:
    attendance_json = json.load(f)
attendance = pd.DataFrame(list(attendance_json.items()), columns=["Name", "Attendance"])

extra = pd.read_excel("extra.xlsx")

print("Students:\n", students)
print("\nAttendance:\n", attendance)
print("\nExtra (Bonus):\n", extra)

merged = students.merge(attendance, on="Name").merge(extra, on="Name")
merged = merged[["Name", "Marks", "Attendance", "Bonus"]]
print("\nMerged DataFrame:\n", merged)
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
one_hot = pd.get_dummies(merged, columns=["Name"])
print("\nOne-Hot Encoded (on Name):\n", one_hot)