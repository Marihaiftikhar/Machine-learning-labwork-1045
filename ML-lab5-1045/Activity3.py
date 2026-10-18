import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
# Task 2: Apply K-Means Algorithm
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='KMeans_Cluster',
    palette='viridis',
    s=80
)
plt.title("K-Means Clustering")
plt.show()
dbscan = DBSCAN(eps=8, min_samples=5)

df['DBSCAN_Cluster'] = dbscan.fit_predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='DBSCAN_Cluster',
    palette='viridis',
    s=80
)
plt.title("DBSCAN Clustering")
plt.show()
# Task 3: Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
df['Scaled_KMeans_Cluster'] = kmeans_scaled.fit_predict(X_scaled)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=X['Annual Income (k$)'],
    y=X['Spending Score (1-100)'],
    hue=df['Scaled_KMeans_Cluster'],
    palette='viridis',
    s=80
)
plt.title("K-Means After StandardScaler")
plt.show()
# Task 4: Compare results before and after scaling
print("\nOriginal K-Means Cluster Counts:")
print(df['KMeans_Cluster'].value_counts().sort_index())
print("\nScaled K-Means Cluster Counts:")
print(df['Scaled_KMeans_Cluster'].value_counts().sort_index())
comparison = pd.crosstab(
    df['KMeans_Cluster'],
    df['Scaled_KMeans_Cluster']
)
print("\nCluster Comparison Table:")
print(comparison)
# Task 5: Elbow Method to find the best k
inertia_values = []
for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia_values.append(model.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
# Task 6: Interpret each cluster
print("\nCluster Interpretation (Scaled K-Means):")
cluster_summary = df.groupby('Scaled_KMeans_Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean().round(2)
print(cluster_summary)
for cluster, row in cluster_summary.iterrows():
    income = row['Annual Income (k$)']
    spending = row['Spending Score (1-100)']

    if income >= 60 and spending >= 60:
        meaning = "High income, high spending"
    elif income >= 60 and spending < 40:
        meaning = "High income, low spending"
    elif income < 40 and spending >= 60:
        meaning = "Low income, high spending"
    elif income < 40 and spending < 40:
        meaning = "Low income, low spending"
    else:
        meaning = "Medium or mixed customer group"

    print(f"Cluster {cluster}: {meaning}")

np.random.seed(42)

df['LuckyNumber'] = np.random.randint(0, 101, size=len(df))

print("\nDataset with LuckyNumber:")
print(df[['Annual Income (k$)',
          'Spending Score (1-100)',
          'LuckyNumber']].head())
# K-Means with LuckyNumber (unscaled)
X_lucky = df[
    ['Annual Income (k$)',
     'Spending Score (1-100)',
     'LuckyNumber']
]
kmeans_lucky = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
df['Lucky_Cluster'] = kmeans_lucky.fit_predict(X_lucky)
X_lucky_scaled = StandardScaler().fit_transform(X_lucky)

kmeans_lucky_scaled = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
df['Lucky_Scaled_Cluster'] = kmeans_lucky_scaled.fit_predict(
    X_lucky_scaled
)
print("\nClusters with LuckyNumber:")
print(df['Lucky_Cluster'].value_counts().sort_index())
print("\nClusters with Scaled LuckyNumber:")
print(df['Lucky_Scaled_Cluster'].value_counts().sort_index())
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Lucky_Scaled_Cluster',
    palette='viridis',
    s=80
)
plt.title("K-Means Clusters Including LuckyNumber")
plt.show()
