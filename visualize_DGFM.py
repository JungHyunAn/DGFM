import numpy as np
import matplotlib.pyplot as plt

# Define the 2D manifold in 3D space
def manifold(u, v):
    x = 1.5*u
    y = 1.5*v
    z = 2*np.sin(u) * np.cos(v)
    return x, y, z

# Define a distribution over the 2D manifold (e.g., a Gaussian bump centered at (0, 0))
def distribution(u, v):
    return (np.exp(-(u**2 + v**2) * 10) + np.exp(-(u**2 + (v-1)**2) * 5) + np.exp(-((u+1.5)**2 + (v)**2) * 2) + np.exp(-(((u-0.5)/2)**2 + (v+1.5)**2))) 

# Create a grid
u = np.linspace(-2, 2, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = manifold(U, V)
D = distribution(U, V)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold surface with distribution as color
surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(D), rstride=1, cstride=1,
                       linewidth=0.5, antialiased=True, alpha=0.9, edgecolor='gray')

# Add a color bar for the distribution
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
mappable.set_array(D)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Density')

# Set labels and title
ax.set_title("2D Manifold in 3D Space with Distribution", fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-2.5, 2.5])

ax.view_init(elev=30, azim=150)

plt.tight_layout()
plt.savefig('visualize_DGFM.png')
plt.close()