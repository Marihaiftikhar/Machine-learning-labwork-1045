import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
df = pd.read_csv("mushroomss.csv")

X = df.drop(columns=["habitat"])
y = (df["habitat"] == "w").astype(int)

print("Minority percentage:", round(y.mean() * 100, 2))
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Step 2 and 3: Encode and scale features, then apply SMOTE
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False), X.columns)
])
model = Pipeline([
    ("encode", preprocessor),
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# Step 4: Train KNN with k=5
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN F1:", f1_score(y_test, y_pred))
# Tune k using GridSearchCV
params = {"knn__n_neighbors": [3, 5, 7, 9, 11]}

grid = GridSearchCV(
    model,
    params,
    cv=3,
    scoring="f1",
    n_jobs=1
)
grid.fit(X_train, y_train)
best_pred = grid.predict(X_test)
print("Best k:", grid.best_params_)
print("Tuned Accuracy:", accuracy_score(y_test, best_pred))
print("Tuned F1:", f1_score(y_test, best_pred))
