import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

# Define the 2D manifold in 3D space
def manifold(u, v):
    x = u
    y = v
    z = np.sin(u) * np.cos(v)
    return x, y, z

# Distribution functions
def distribution(u, v):
    return (np.exp(-((u-1)**2 + v**2) * 5) +
            np.exp(-(u**2 + (v-1)**2) * 5) +
            np.exp(-((u+0.25)**2 + (v+1)**2) * 2)) * 0.9

# Create grid
u = np.linspace(0, 2, 100)
v = np.linspace(-1, 1, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = manifold(U, V)
D = distribution(U, V)

# Sample points from the first Gaussian bump
D_comp = np.exp(-((U-1)**2 + V**2) * 5)
probs = D_comp.flatten() / D_comp.flatten().sum()
indices = np.random.choice(U.size, size=50, p=probs)
sample_u = U.flatten()[indices]
sample_v = V.flatten()[indices]
X_s, Y_s, Z_s = manifold(sample_u, sample_v)
samples = np.stack([X_s, Y_s, Z_s], axis=1)

# PCA for plane
pca = PCA(n_components=2)
scores = pca.fit_transform(samples)
mean_3d = pca.mean_
axis1, axis2 = pca.components_

# Compute covariance in PCA space and ellipse parameters
cov_pca = np.cov(scores, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_pca)
# Radii for 2 standard deviations
radii = 2 * np.sqrt(eigvals)

# Parametric ellipse in PCA coords
phi = np.linspace(0, 2*np.pi, 200)
ellipse_2d = np.vstack((radii[0] * np.cos(phi), radii[1] * np.sin(phi)))
ellipse_rotated = eigvecs @ ellipse_2d  # rotate by eigenvectors

# Map ellipse to 3D
ellipse_3d = mean_3d[:, None] + axis1[:, None] * ellipse_rotated[0] + axis2[:, None] * ellipse_rotated[1]

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Manifold surface with original distribution colors
surf = ax.plot_surface(
    X, Y, Z,
    facecolors=plt.cm.viridis(D / D.max()),
    rstride=1, cstride=1,
    linewidth=0.5, antialiased=True,
    alpha=0.9, edgecolor='gray'
)

# Gaussian plane colored differently (using plasma colormap)
# Project the Gaussian PDF grid back to 3D plane
grid_size = 50
x_lin = np.linspace(scores[:, 0].min() - 0.5, scores[:, 0].max() + 0.5, grid_size)
y_lin = np.linspace(scores[:, 1].min() - 0.5, scores[:, 1].max() + 0.5, grid_size)
Xg, Yg = np.meshgrid(x_lin, y_lin)
pos = np.dstack((Xg, Yg))
rv = multivariate_normal(scores.mean(axis=0), cov_pca)
Zg = rv.pdf(pos)
Zg_norm = (Zg - Zg.min()) / (Zg.max() - Zg.min())

# Map grid to 3D
Xp = mean_3d[0] + axis1[0] * Xg + axis2[0] * Yg
Yp = mean_3d[1] + axis1[1] * Xg + axis2[1] * Yg
Zp = mean_3d[2] + axis1[2] * Xg + axis2[2] * Yg

plane = ax.plot_surface(
    Xp, Yp, Zp,
    facecolors=plt.cm.spring(Zg_norm),
    rstride=1, cstride=1,
    linewidth=0, antialiased=False,
    alpha=0.3
)

# Plot ellipse as 3D line
ax.plot(ellipse_3d[0], ellipse_3d[1], ellipse_3d[2],
        color='red', linewidth=2, label='Gaussian Ellipse (2σ)')

# Plot samples in red
ax.scatter(X_s, Y_s, Z_s, color='red', s=50, depthshade=True, label='Samples')

# Plot principal axes
scale = radii.max()
ax.quiver(
    mean_3d[0], mean_3d[1], mean_3d[2],
    axis1[0]*scale, axis1[1]*scale, axis1[2]*scale,
    color='black', linewidth=2, arrow_length_ratio=0.1, label='PC1'
)
ax.quiver(
    mean_3d[0], mean_3d[1], mean_3d[2],
    axis2[0]*scale, axis2[1]*scale, axis2[2]*scale,
    color='gray', linewidth=2, arrow_length_ratio=0.1, label='PC2'
)

# Color bar
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(D)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Density')

# Labels and view
ax.set_title("3D Manifold with PCA Gaussian Ellipsoid and Samples", fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=120)
ax.legend()

plt.tight_layout()
plt.show()
