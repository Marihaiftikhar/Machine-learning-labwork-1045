import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

feature_cols = list(data.feature_names)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])
print("Feature scaling applied. Distance-based models (KNN, SVM) compute distances "
      "between points directly, so unscaled features with larger ranges would "
      "dominate the distance calculation and bias the model.")