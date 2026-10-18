import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
df = pd.read_csv("income_evaluation.csv")
# Remove extra spaces from column names
df.columns = df.columns.str.strip()
# Separate features and target
X = df.drop("income", axis=1)
y = df["income"].astype(str).str.strip()
# Handle missing values
X = X.replace("?", pd.NA)

for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in X.select_dtypes(include="number").columns:
    X[col] = X[col].fillna(X[col].median())

# Convert categorical features to numeric
X = pd.get_dummies(X, dtype=int)

# Step 2: Split data (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Step 3: StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 4: Train SVM
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)
# Step 5: Prediction
y_pred = model.predict(X_test)
# Step 6: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
# Step 7: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Step 8: Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    cmap="Blues"
)

plt.title("SVM Confusion Matrix")
plt.show()
# Step 9: Display actual and predicted labels
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})
print("\nSample Predictions:")
print(results.head(10))