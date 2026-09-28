import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df["mean area"], bins=30)
plt.title("Mean Area - Before Log Transform")

df["log_mean_area"] = np.log1p(df["mean area"])

plt.subplot(1, 2, 2)
plt.hist(df["log_mean_area"], bins=30)
plt.title("Mean Area - After Log Transform")
plt.tight_layout()
plt.show()