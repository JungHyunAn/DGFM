# Modified script to generate 'visualize_DGFM_with_Gaussian.png'
# Adds 1000 samples from a 3D standard Gaussian overlaid on the original manifold figure
# and makes the manifold slightly transparent.

import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
rng = np.random.default_rng(42)

# Define the 2D manifold in 3D space
def manifold(u, v):
    x = 1.5*u
    y = 1.5*v
    z = 2*np.sin(u) * np.cos(v)
    return x, y, z

# Define a distribution over the 2D manifold (e.g., a Gaussian bump centered at (0, 0))
def distribution(u, v):
    return (
        np.exp(-(u**2 + v**2) * 10)
        + np.exp(-(u**2 + (v - 1) ** 2) * 5)
        + np.exp(-((u + 1.5) ** 2 + (v) ** 2) * 2)
        + np.exp(-(((u - 0.5) / 2) ** 2 + (v + 1.5) ** 2))
    )

# Create a grid
u = np.linspace(-2, 2, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = manifold(U, V)
D = distribution(U, V)

# Normalize D for coloring (optional but makes colormap robust)
D_norm = (D - D.min()) / (D.max() - D.min() + 1e-12)

# Sample 1000 points from a standard 3D Gaussian
num_samples = 1500
gauss_samples = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 3))
gx, gy, gz = gauss_samples[:, 0], gauss_samples[:, 1], gauss_samples[:, 2]

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold surface with distribution as color; slightly transparent
surf = ax.plot_surface(
    X, Y, Z,
    facecolors=plt.cm.viridis(D_norm),
    rstride=1, cstride=1,
    linewidth=0.2, antialiased=True,
    alpha=0.3, edgecolor='none'
)

# Overlay the Gaussian samples
ax.scatter(gx, gy, gz, s=8, depthshade=True, alpha=0.9)

# Add a color bar for the distribution
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(D)  # use original density scale for the colorbar
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Density')

# Set labels and title
ax.set_title("2D Manifold in 3D with Overlaid Standard Gaussian Samples", fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-2.5, 2.5])

# Nice viewing angle
ax.view_init(elev=30, azim=150)

plt.tight_layout()
out_path = 'visualize_DGFM_with_Gaussian.png'
plt.savefig(out_path)
plt.close()