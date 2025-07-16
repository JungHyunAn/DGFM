import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
n = 10000
branch_range = np.arange(-4, 5)  # -4 … +4
num_k = branch_range.size

# Create grid on [0, 2π]²
x = np.linspace(0, 2*np.pi, n)
y = np.linspace(0, 2*np.pi, n)
X, Y = np.meshgrid(x, y)
S = X + Y

# Compute rhs and mask invalid points
rhs = 1 - (np.sin(X) + np.sin(S))
valid = np.abs(rhs) <= 1

# Initialize base branches
Z1_base = np.full_like(rhs, np.nan)
Z2_base = np.full_like(rhs, np.nan)

# Only fill where valid
Z1_base[valid] = np.arcsin(rhs[valid]) - S[valid]
Z2_base[valid] = np.pi - np.arcsin(rhs[valid]) - S[valid]

# Prepare colormaps
colors_z1 = cm.spring(np.linspace(0, 1, num_k))
colors_z2 = cm.summer(np.linspace(0, 1, num_k))

# Set up figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each branch
for i, k in enumerate(branch_range):
    # Shift branches by 2π·k
    Z1 = Z1_base + 2*np.pi*k
    Z2 = Z2_base + 2*np.pi*k

    # Mask outside [0, 2π] by same test as MATLAB: abs(z - π) > π
    mask1 = np.abs(Z1 - np.pi) > np.pi
    mask2 = np.abs(Z2 - np.pi) > np.pi
    Z1[mask1] = np.nan
    Z2[mask2] = np.nan

    # Plot with flat color and no edges
    ax.plot_surface(X, Y, Z1,
                    color=colors_z1[i],
                    edgecolor='none',
                    shade=False,
                    alpha=0.8,
                    rcount=200, ccount=200)
    ax.plot_surface(X, Y, Z2,
                    color=colors_z2[i],
                    edgecolor='none',
                    shade=False,
                    alpha=0.8,
                    rcount=200, ccount=200)

# Axes limits and labels
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(0, 2*np.pi)
ax.set_zlim(0, 2*np.pi)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(r"Manifold from $\sin(x) + \sin(x+y) + \sin(x+y+z) = 1$")

ax.view_init(elev=25, azim=150)

plt.savefig('robot_example.png')
plt.close()