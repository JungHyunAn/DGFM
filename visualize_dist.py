# Re-import libraries due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_A(shape):
    return np.random.normal(0, 1 / np.sqrt(shape[1]), size=shape)

def sample_Q(n, d):
    return np.random.normal(0, 1 / np.sqrt(n * d), size=(n, d, d))

def quadratic_term(Q, z):
    return np.einsum('ndk,bk,bj->bn', Q, z, z)

def plot_points_and_manifold(samples, manifold_points, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(manifold_points[:, 0], manifold_points[:, 1], manifold_points[:, 2], s=1, alpha=0.2, label='Manifold')
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=5, color='red', label='Samples')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_and_plot_distributions(n=3, d=2, num_samples=500, epsilon=0.1, seed=0):
    np.random.seed(seed)

    # Quadratic Unimodal
    A = sample_A((n, d))
    Q = sample_Q(n, d)
    z = np.random.randn(num_samples, d)
    x_n = np.random.randn(num_samples, n)
    x = z @ A.T + quadratic_term(Q, z) + epsilon * x_n

    grid = np.linspace(-3.5, 3.5, 100)
    zz1, zz2 = np.meshgrid(grid, grid)
    zz = np.stack([zz1.ravel(), zz2.ravel()], axis=-1)
    manifold = zz @ A.T + quadratic_term(Q, zz)
    plot_points_and_manifold(x, manifold, "Quadratic Unimodal", "Quadratic_Unimodal_demo.png")

    # Quadratic Multimodal
    m = 3
    mu = np.random.randn(m, d) * 2
    # mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    samples_per_mode = num_samples // m
    z_list = [np.random.randn(samples_per_mode, d)/3 + mu[i] for i in range(m)]
    z = np.vstack(z_list)
    remaining = num_samples - z.shape[0]
    if remaining > 0:
        z = np.vstack([z, np.random.randn(remaining, d) + mu[0]])
    x_n = np.random.randn(num_samples, n)
    x = z @ A.T + quadratic_term(Q, z) + epsilon * x_n
    manifold = zz @ A.T + quadratic_term(Q, zz)
    plot_points_and_manifold(x, manifold, "Quadratic Multimodal", "Quadratic_Multimodal_demo.png")

    # Linear Branched
    t = 2
    basis = sample_A((n, n))
    B_list = [basis[:, np.random.choice(n, d, replace=False)] for _ in range(t)]
    z = np.random.randn(num_samples, d)
    x_n = np.random.randn(num_samples, n)
    branch_idx = np.random.choice(t, num_samples)
    x = np.array([z[i] @ B_list[branch_idx[i]].T + epsilon * x_n[i] for i in range(num_samples)])

    manifold = []
    for B in B_list:
        manifold.append(zz @ B.T)
    manifold = np.concatenate(manifold, axis=0)
    plot_points_and_manifold(x, manifold, "Linear Branched", "Linear_Branched_demo.png")

    # Generalized Swiss Roll
    A = sample_A((n, n))
    s = 2
    z1 = np.random.uniform(0, 2 * s * np.pi, size=(num_samples, 1))
    z_rest = np.random.randn(num_samples, d - 1)
    z = np.hstack([z1, z_rest])
    spiral = np.hstack([
        z1 * np.cos(z1),
        z1 * np.sin(z1),
        z_rest,
        np.zeros((num_samples, n - d - 1))
    ])
    x_n = np.random.randn(num_samples, n)
    x = spiral @ A.T + epsilon * x_n

    theta = np.linspace(0, 2 * s * np.pi, 10000)
    r = np.stack([theta * np.cos(theta), theta * np.sin(theta)], axis=1)
    z_rest_manifold = np.random.uniform(-2, 2, size=(10000, d - 1)) * 3
    spiral_manifold = np.hstack([
        r, z_rest_manifold, np.zeros((10000, n - d - 1))
    ])
    manifold = spiral_manifold @ A.T
    plot_points_and_manifold(x, manifold, "Generalized Swiss Roll", "Generalized_Swiss_Roll_demo.png")

# Run the visualization with manifold
generate_and_plot_distributions(seed=11)
