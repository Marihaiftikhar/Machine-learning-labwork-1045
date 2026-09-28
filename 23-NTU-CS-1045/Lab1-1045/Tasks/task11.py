import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[list(data.feature_names)])

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, df["target"], test_size=0.2, random_state=42
)

svm_model = SVC().fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

knn_model = KNeighborsClassifier().fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"KNN Accuracy: {knn_acc:.4f}")