from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_cleaning import clean_data

# Load and clean the data
file_path = 'transfusion.csv'
df = clean_data(file_path)

# Prepare features
X = df[['Recency', 'Frequency', 'Monetary', 'Time']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)  # You can experiment with n_clusters
kmeans.fit(X_scaled)

# Print the cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centers')
plt.title('KMeans Clustering (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score

# Inertia (already available in kmeans model)
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")

# Silhouette Score
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {sil_score}")
from sklearn.metrics import silhouette_score

for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"n_clusters={n_clusters}, Silhouette Score={sil_score}")
