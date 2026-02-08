import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from functools import lru_cache
import multiprocessing as mp
import time
from matplotlib.colors import LinearSegmentedColormap

N = 6; M = 2; M1 = 4; b = 1; T = 1; c = 0.1; r = 0.5; a1 = 0.03; a2 = 0.05; cp = 0.06; omega = 4/5;
F = 1 / (1 - omega); s = 2






@lru_cache(maxsize=None)
def binom_coeff(n, k):
    return binom(n, k)



def precompute_coeffs():
    coeffs = []
    for i in range(0, N):
        fuzhuk = N - 1 - i
        for j in range(0, fuzhuk + 1):
            k = N - 1 - i - j
            comb = binom_coeff(N - 1, i) * binom_coeff(N - 1 - i, j)
            theta1 = 1 if (i + 1) >= M else 0
            theta2 = 1 if i >= M else 0
            theta3 = 1 if k >= T else 0
            theta4 = 1 if (i + 1) >= M1 else 0
            theta5 = 1 if i >= M1 else 0
            coeffs.append((i, j, k, comb, theta1, theta2, theta3, theta4, theta5))
    return coeffs



precomputed_coeffs = precompute_coeffs()


def calculate_payoffs_vectorized(x, z):
    """向量化计算三种策略的期望收益"""
    y = 1 - x - z


    Pc = 0
    Pd = 0
    Pp = 0


    for i, j, k, comb, theta1, theta2, theta3, theta4, theta5 in precomputed_coeffs:

        if x < 1e-10 and i > 0:
            continue
        if y < 1e-10 and j > 0:
            continue
        if z < 1e-10 and k > 0:
            continue

        weight = comb * (x ** i) * (y ** j) * (z ** k)

        Pc += weight * ((-c + b * theta1 + (1 - r) * b * (1 - theta1)) * (s - 1) + (theta1 * (-c + b * theta4 + (1 - r) * (1 - theta4) * b)+ (1 - theta1) * (-c + b * theta1 + (1 - r) * b * (1-theta1))) * (F - s + 1))
        Pd += weight * ((b * theta2 + (1 - r) * b * (1 - theta2) - (a1 * theta3)) * (s - 1) + (theta2 * (b * theta5 + (1 -r) * b * (1-theta5) - a1 * theta3) + (1 - theta2) * (b * theta2 + (1 - r) * b * (1 - theta2) - a2 * theta3)) * (F - s + 1))
        Pp += weight * ((-cp + b * theta2 + (1 - r) * b * (1 - theta2)) * (s - 1) + (theta2 * (-cp + b * theta5 + (1 - r) * b * (1-theta5)) + (1 - theta2) * (-cp + b * theta2 + (1 - r) * b *(1 - theta2))) * (F - s + 1))

    return Pc, Pd, Pp


def replicator_dynamics(x, z):
    """计算复制动态方程"""
    if x < 0 or z < 0 or x + z > 1:
        return 0, 0, 0

    Pc, Pd, Pp = calculate_payoffs_vectorized(x, z)


    P_avg = x * Pc + (1 - x - z) * Pd + z * Pp


    dx = x * (Pc - P_avg)
    dz = z * (Pp - P_avg)


    mag = np.sqrt(dx ** 2 + dz ** 2)

    return dx, dz, mag


def process_point(args):
    """处理单个点的函数，用于并行计算"""
    i, j, x_val, z_val = args


    x_simplex = 2 * z_val / np.sqrt(3)
    z_simplex = x_val - z_val / np.sqrt(3)

    


    if (z_simplex >= 0) and (x_simplex >= 0) and (x_simplex + z_simplex <= 1):
        dx, dz, mag = replicator_dynamics(x_simplex, z_simplex)


        if mag > 1e-10:
            dx /= mag
            dz /= mag


        u_val = 1/2 * dx + dz
        v_val = (np.sqrt(3) / 2) * dx
        return (i, j, u_val, v_val, mag)

    return (i, j, np.nan, np.nan, 0)


if __name__ == '__main__':
    n_points = 100
    x_range = np.linspace(0, 1, n_points)
    z_range = np.linspace(0, np.sqrt(3) / 2, n_points)
    X, Z = np.meshgrid(x_range, z_range)


    U = np.zeros_like(X)
    V = np.zeros_like(X)
    Magnitude = np.zeros_like(X)


    print(f"Using {mp.cpu_count()} CPU cores for parallel computation...")
    start_time = time.time()


    args_list = []
    for i in range(n_points):
        for j in range(n_points):
            args_list.append((i, j, X[i, j], Z[i, j]))

    with mp.Pool(mp.cpu_count()) as pool:

        batch_size = 1000
        results = []
        for idx in range(0, len(args_list), batch_size):
            batch = args_list[idx:idx + batch_size]
            batch_results = pool.map(process_point, batch)
            results.extend(batch_results)


    for i, j, u_val, v_val, mag in results:
        U[i, j] = u_val
        V[i, j] = v_val
        Magnitude[i, j] = mag

    print(f"Computation completed! Time elapsed: {time.time() - start_time:.2f} seconds")


    max_mag = np.nanmax(Magnitude)
    min_mag = np.nanmin(Magnitude)

    if max_mag > min_mag:
        Magnitude_normalized = (Magnitude - min_mag) / (max_mag - min_mag)
        print(f"Magnitude range: {min_mag:.4f} - {max_mag:.4f}")
        print(
            f"Normalized magnitude range: {np.nanmin(Magnitude_normalized):.4f} - {np.nanmax(Magnitude_normalized):.4f}")
    else:
        Magnitude_normalized = np.zeros_like(Magnitude)
        print("All magnitude values are identical or invalid")


    plt.figure(figsize=(6, 4.4))

    colors = [
        '#264653',
        '#2A9D8F',
        '#E9C46A',
        '#F4A261',
        '#E76F51'
    ]


    cmap1 = LinearSegmentedColormap.from_list('deep_spectrum', colors, N=256)


    strm = plt.streamplot(X, Z, U, V,
                          density=1.0,
                          color=Magnitude_normalized,
                          cmap=cmap1,
                          arrowsize=1.5,
                          linewidth=1.5,
                          broken_streamlines=True
                          )


    cbar = plt.colorbar(strm.lines)

    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], 'k-', lw=2)

    plt.text(-0.02, -0.01, 'D', fontsize=12, ha='center')
    plt.text(1.02, -0.01, 'P', fontsize=12, ha='center')
    plt.text(0.5, 0.9, 'C', fontsize=12, ha='center')

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 0.95)
    plt.axis('off')
    plt.title('', fontsize=14)
    plt.tight_layout()
    plt.show()