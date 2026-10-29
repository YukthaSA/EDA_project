import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"

# PART A : K-MEANS CLUSTERING

print("\nK-MEANS CLUSTERING\n")

# 1. Load dataset
df = pd.read_csv(URL)
print("1. Loaded:", df.shape, "\n")

# 2. View dataset
print("2. Dataset preview\n", df.head(), "\n")

# 3. Missing values
print("3. Missing values:", int(df.isna().sum().sum()), "\n")

# 4. Dependent variable distribution (table)
print("4. Attrition distribution\n", df["Attrition"].value_counts(), "\n")

# 5. Numerical features only (others removed).
#    Zero-variance columns (EmployeeCount, StandardHours) carry no information
#    and cannot be standardised, so they are removed too.
X = df.select_dtypes(include="number")
X = X.loc[:, X.std() > 0]
print("5. Numerical features used:", X.shape[1], "\n   ", list(X.columns), "\n")

# 6. Standardise (scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("6. Standardised data  mean ~", X_scaled.mean().round(3), " std ~", X_scaled.std().round(3), "\n")

# 7. K-Means with 2 centres
km2 = KMeans(n_clusters=2, n_init=25, random_state=42)
km2.fit(X_scaled)

# 8. Display K-Means results
print("8. K-Means (k=2) results")
print("   Cluster sizes:", np.bincount(km2.labels_))
print("   Inertia (total within-cluster SS):", round(km2.inertia_, 2))
print("   First 10 assignments:", km2.labels_[:10])
print("   Cross-tab with Attrition\n", pd.crosstab(km2.labels_, df["Attrition"]), "\n")

# 9. Cluster centres
centers = pd.DataFrame(km2.cluster_centers_, columns=X.columns)
print("9. Cluster centres (standardised scale)\n", centers.round(3).T, "\n")

# 10. WCSS for K = 1 to 10
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X_scaled)
    wcss.append(km.inertia_)
print("10. WCSS values")
for k, w in zip(range(1, 11), wcss):
    print(f"    K={k:2d}  WCSS={w:10.2f}")
print()

# 11. Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o", color="blue")
plt.title("11. Elbow Method"); plt.xlabel("Number of clusters (K)"); plt.ylabel("WCSS")
plt.xticks(range(1, 11)); plt.grid(True, alpha=0.4)
plt.show()

# 12. K-Means at the elbow recommendation
ks = np.arange(1, 11)
p1, p2 = np.array([1, wcss[0]]), np.array([10, wcss[-1]])
pts = np.column_stack([ks, wcss])
v = p2 - p1
dist = np.abs(v[0] * (pts[:, 1] - p1[1]) - v[1] * (pts[:, 0] - p1[0])) / np.linalg.norm(v)
best_k = int(ks[dist.argmax()])
print("12. Elbow recommendation: K =", best_k)

km_best = KMeans(n_clusters=best_k, n_init=25, random_state=42).fit(X_scaled)
print("    Cluster sizes:", np.bincount(km_best.labels_))
print("    Cross-tab with Attrition\n", pd.crosstab(km_best.labels_, df["Attrition"]), "\n")

# Visualise the clusters on the first two features
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, X.columns.get_loc("MonthlyIncome")],
            c=km_best.labels_, cmap="viridis", alpha=0.6)
plt.xlabel("Age (scaled)"); plt.ylabel("MonthlyIncome (scaled)")
plt.title(f"12. K-Means clusters (K = {best_k})")
plt.show()

# PART B : HIERARCHICAL CLUSTERING

print("\nHIERARCHICAL CLUSTERING\n")

# 1. Load dataset
hdf = pd.read_csv(URL)
print("1. Loaded:", hdf.shape, "\n")

# 2. View dataset
print("2. Dataset preview\n", hdf.head(), "\n")

# 3. Missing values
print("3. Missing values:", int(hdf.isna().sum().sum()), "\n")

# 4. Numerical attributes only
HX = hdf.select_dtypes(include="number")
HX = HX.loc[:, HX.std() > 0]
print("4. Numerical attributes:", HX.shape[1], "\n")

# 5. Standardise
HX_scaled = StandardScaler().fit_transform(HX)

# For readability and speed the dendrogram is built on a random sample of 150 employees
rng = np.random.default_rng(1)
idx = rng.choice(len(HX_scaled), 150, replace=False)
H = HX_scaled[idx]
labels_sample = hdf["EmployeeNumber"].values[idx]
print("5. Standardised; using a random sample of", len(idx), "employees for the dendrogram\n")

# 6. Euclidean distance
d = pdist(H, metric="euclidean")
print("6. Euclidean distances computed:", len(d), "pairs\n")

# 7. Hierarchical clustering with Ward's method
Z = linkage(H, method="ward", metric="euclidean")
print("7. Ward linkage matrix computed:", Z.shape, "\n")

# 8. Dendrogram
plt.figure(figsize=(14, 6))
dendrogram(Z, labels=labels_sample, leaf_rotation=90, leaf_font_size=6)
plt.title("8. Dendrogram (Ward's method)"); plt.xlabel("EmployeeNumber"); plt.ylabel("Distance")
plt.tight_layout(); plt.show()

# 9. Cut into 3 clusters + assignments
clusters = fcluster(Z, t=3, criterion="maxclust")
assign = pd.DataFrame({"EmployeeNumber": labels_sample, "Cluster": clusters})
print("9. Cluster assignments (first 15)\n", assign.head(15), "\n")

# 10. Number of samples per cluster
print("10. Samples in each cluster\n", pd.Series(clusters).value_counts().sort_index())
