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

def generate_and_plot_distributions(n=3, d=2, num_samples=500, epsilon=0.05, seed=0):
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

    # Generalized Two Moons
    A = sample_A((n, n))
    z1 = np.random.uniform(0, np.pi, size=(num_samples, 1))
    z_rest = np.random.randn(num_samples, d - 1)

    # First moon (as requested)
    moon1 = np.hstack([
        np.cos(z1) + 0.5,
        np.sin(z1),
        z_rest,
        np.zeros((num_samples, n - d - 1))
    ])

    # Second moon: symmetric to O (origin), i.e., x -> -x
    moon2 = -moon1

    # Combine two moons
    moon = np.vstack([moon1, moon2])

    # Add noise and random linear mixing
    x_n = np.random.randn(2 * num_samples, n)
    x = moon @ A.T + epsilon * x_n

    # ---- Manifold for visualization (dense points on both moons) ----
    theta = np.linspace(0, np.pi, 10000).reshape(-1, 1)
    r = np.hstack([
        np.cos(theta) + 0.5,
        np.sin(theta)
    ])

    # Use a bounded spread for the remaining intrinsic dims (match your Swiss-roll style)
    z_rest_manifold = (np.random.uniform(-2, 2, size=(10000, d - 1)) * 3)

    moon1_manifold = np.hstack([
        r,
        z_rest_manifold,
        np.zeros((10000, n - d - 1))
    ])

    moon2_manifold = -moon1_manifold

    moon_manifold = np.vstack([moon1_manifold, moon2_manifold])
    manifold = moon_manifold @ A.T

    plot_points_and_manifold(
        x, manifold,
        "Generalized Two Moon",
        "Generalized_Two_Moon_demo.png"
    )


    # PinWheel
    # Random linear mixing
    wheel_num = 3
    A = sample_A((n, n))

    # ---- Sample points x (like PinWheel.sample()) ----
    t = np.random.rand(num_samples)  # t ~ U(0,1)

    # wheel indices i ~ Unif{0..wheel_num-1}
    wheel_idx = np.random.randint(0, wheel_num, size=(num_samples,))

    angles = (2.0 * np.pi / wheel_num) * wheel_idx
    c = np.cos(angles)
    s = np.sin(angles)

    base0 = t
    base1 = np.sin(np.pi * t)  # IMPORTANT: sin(pi*t), matching your class

    v = np.zeros((num_samples, n))

    # Rotated curve in first 2 dims
    v[:, 0] = c * base0 - s * base1
    v[:, 1] = s * base0 + c * base1

    # Remaining intrinsic dims: Gaussian for samples (matches class)
    if d > 1:
        z_rest = np.random.randn(num_samples, d - 1)
        v[:, 2:d + 1] = z_rest  # dims 2..d

    # Apply linear mixing and add noise
    x_n = np.random.randn(num_samples, n)
    x = v @ A.T + epsilon * x_n


    # ---- Manifold for visualization (dense points on each wheel) ----
    N_man = 3000
    t_dense = np.linspace(0.0, 1.0, N_man)  # parameter along the curve

    # bounded spread for remaining intrinsic dims (Swiss-roll style)
    # (If you want it to reflect the *sample* distribution more, switch to np.random.randn)
    z_rest_manifold = (np.random.uniform(-2, 2, size=(N_man, d - 1)) * 3) if d > 1 else None

    base0_m = t_dense
    base1_m = np.sin(np.pi * t_dense)

    manifold_list = []
    for i in range(wheel_num):
        angle = (2.0 * np.pi / wheel_num) * i
        ci = np.cos(angle)
        si = np.sin(angle)

        v_m = np.zeros((N_man, n))
        v_m[:, 0] = ci * base0_m - si * base1_m
        v_m[:, 1] = si * base0_m + ci * base1_m

        if d > 1:
            v_m[:, 2:d + 1] = z_rest_manifold

        manifold_list.append(v_m)

    v_manifold = np.vstack(manifold_list)      # (wheel_num*N_man, n)
    manifold = v_manifold @ A.T                # mixed manifold in ambient space


    plot_points_and_manifold(
        x, manifold,
        "Generalized Pin Wheel",
        "Generalized_PinWheel_demo.png"
    )

# Run the visualization with manifold
generate_and_plot_distributions(seed=23)
