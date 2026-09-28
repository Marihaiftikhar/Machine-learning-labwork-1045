import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.utils.class_weight import compute_class_weight

data = load_breast_cancer()
print(data.keys())
print(data.target_names)

n_samples, n_features = data.data.shape
print('Number of samples:', n_samples)
print('Number of features:', n_features)
print('Feature Names:', data.feature_names)
print("Dimension of Input:", data.data.shape)
print("Dimension of Output:", data.target.shape)

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("\nMissing values in df (before):", df.isnull().sum().sum())

np.random.seed(42)
random_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[random_idx, "mean radius"] = np.nan
print("Missing values in 'mean radius' after artificial introduction:",
      df["mean radius"].isnull().sum())
df["mean radius"] = df["mean radius"].fillna(df["mean radius"].median())
print("Missing values in 'mean radius' after handling:", df["mean radius"].isnull().sum())

counts_before = df["target"].value_counts()
print("\nClass counts (0=malignant, 1=benign):\n", counts_before)

classes = np.unique(df["target"])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["target"])
class_weight_dict = dict(zip(classes, weights))
print("\nDataset is mildly imbalanced (~37% malignant vs ~63% benign).")
print("Class weighting is applied (instead of SMOTE) to avoid extra resampling libraries:")
print("Computed class weights:", class_weight_dict)

df.to_csv("activity6_cleaned.csv", index=False)