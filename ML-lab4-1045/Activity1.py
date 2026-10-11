
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 1. Load dataset
data = pd.read_csv("creditcard.csv")

print(data.head())
print(data["Class"].value_counts())

# 2. Split features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("under", RandomUnderSampler(
        sampling_strategy=0.1, random_state=42
    )),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=1))
])

# 5. Train KNN with k=5
print("\nTraining KNN with k=5...")
pipeline.fit(X_train, y_train)

# 6. Predict
print("Making predictions...")
y_pred = pipeline.predict(X_test)

# 7. Evaluate initial model
print("\nKNN Model (k=5)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# 8. Tune k using GridSearchCV
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=1
)

print("\nTuning k...")
grid_search.fit(X_train, y_train)

# 9. Show best k
print("\nBest k:", grid_search.best_params_["knn__n_neighbors"])
print("Best CV F1-Score:", grid_search.best_score_)

# 10. Evaluate tuned model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

print("\nTuned KNN Model")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("F1-Score:", f1_score(y_test, y_pred_tuned))

# 11. Compare results
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "F1-Score"],
    "KNN (k=5)": [
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ],
    "Tuned KNN": [
        accuracy_score(y_test, y_pred_tuned),
        f1_score(y_test, y_pred_tuned)
    ]
})

print("\nModel Comparison")
print(comparison)
