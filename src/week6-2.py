import matplotlib.pyplot as plt
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pl.read_csv("data/penguins.csv")
columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

df = df.drop_nulls()

X = df.select(columns).to_numpy()
y = df.select("species").to_series().to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
colors = ["red", "blue", "green"]
labels = df["species"].unique()

for i, label in enumerate(labels):
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], c=colors[i], label=label)

ax.set_xlabel(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
ax.set_ylabel(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
ax.set_zlabel(f"PC3: {pca.explained_variance_ratio_[2]:.3f}")  # type: ignore
ax.legend()

explained_variance_ratio = pca.explained_variance_ratio_

plt.savefig("outputs/penguin_pca_plot.png", dpi=300)
plt.show()
