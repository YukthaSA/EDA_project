import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from adjustText import adjust_text        
except ImportError:
    adjust_text = None
    print("Tip: run  pip install adjustText  for automatically non-overlapping labels.\n")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/Employee%20Attrition.csv"
REG_NO = "23BDS0287"          

# 1. Load dataset (libraries pandas, numpy, matplotlib, sklearn are imported above)
df = pd.read_csv(URL)
print("1. Dataset loaded; packages imported (pandas, numpy, matplotlib, scikit-learn).\n")

# 2. Rows and columns
print(f"2. Total rows = {df.shape[0]}, total columns = {df.shape[1]}\n")

# 3. Numeric columns -> clean -> normalise  (data_scaled)
num = df.select_dtypes(include="number").dropna()        
num = num.loc[:, num.std() > 0]                           
data_scaled = pd.DataFrame(StandardScaler().fit_transform(num), columns=num.columns)
print("3. data_scaled shape:", data_scaled.shape, "\n")

# 4. PCA + summary
pca = PCA()
scores = pca.fit_transform(data_scaled)
n_pc = scores.shape[1]
pc_names = [f"PC{i}" for i in range(1, n_pc + 1)]
summary = pd.DataFrame({
    "Standard deviation": np.sqrt(pca.explained_variance_),
    "Proportion of Variance": pca.explained_variance_ratio_,
    "Cumulative Proportion": np.cumsum(pca.explained_variance_ratio_),
}, index=pc_names).round(4)
print("4. PCA summary\n", summary, "\n")

# 5. Scree plot with data labels, y-axis 0-60
var_pct = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(11, 5))
plt.plot(range(1, n_pc + 1), var_pct, marker="o", color="steelblue")
for i, v in enumerate(var_pct, start=1):
    plt.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
plt.ylim(0, 60)
plt.xticks(range(1, n_pc + 1))
plt.title("5. Scree Plot"); plt.xlabel("Principal Component"); plt.ylabel("Percentage of Variance Explained")
plt.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()

# 6. PCA biplot (observations + variable loading arrows, non-overlapping labels)
loadings = pca.components_[:2].T                         # (variables x 2)
scale = np.abs(scores[:, :2]).max() * 0.8
fig, ax = plt.subplots(figsize=(11, 9))
ax.scatter(scores[:, 0], scores[:, 1], s=8, alpha=0.25, color="grey")
texts = []
for i, var in enumerate(num.columns):
    x, y = loadings[i, 0] * scale, loadings[i, 1] * scale
    ax.arrow(0, 0, x, y, color="red", alpha=0.7, head_width=0.08, length_includes_head=True)
    texts.append(ax.text(x, y, var, color="darkred", fontsize=9))
if adjust_text:
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="darkred", lw=0.5))
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)")
ax.set_title("6. PCA Biplot - Employee Attrition Data")
plt.tight_layout(); plt.show()

# 7. PCA scores -> data frame pca_scores, first six rows
pca_scores = pd.DataFrame(scores, columns=pc_names)
print("7. First six rows of pca_scores\n", pca_scores.head(6), "\n")

# 8. Scatter plot PC1 vs PC2 : blue circular points, registration number in title
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores["PC1"], pca_scores["PC2"], color="blue", marker="o", alpha=0.6)
plt.title(f"8. PC1 vs PC2 - {REG_NO}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()
