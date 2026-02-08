import numpy as np
from scipy.special import comb
from functools import lru_cache
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from joblib import Parallel, delayed
from matplotlib import colormaps
from scipy.interpolate import griddata


Z = 100; N = 6; M = 2; M1 = 4; T = 1; c = 0.1; b = 1; r = 0.2; a1 = 0.03;
a2 = 0.3; cp = 11/250; mu = 0.01; omega = 0.8; s = 2; F = 1 / (1 - omega); beta = 5





@lru_cache(maxsize=None)
def binom_coeff(n, k):
    """计算二项式系数C(n, k)"""
    if k < 0 or k > n:
        return 0
    return comb(n, k, exact=True)



HDdata = np.zeros((Z + 1, Z + 1))
for i in range(Z + 1):
    for j in range(Z + 1):
        if j <= i:
            HDdata[i, j] = binom_coeff(i, j)



def prob_D(N_C, N_P, i_C, i_P, Z, N, HDdata):
    if N_C > i_C or N_P > i_P or (N - N_C - N_P - 1) > (Z - i_C - i_P - 1):
        return 0
    num = HDdata[i_C, N_C] * HDdata[i_P, N_P] * HDdata[Z - i_C - i_P - 1, N - N_C - N_P - 1]
    denom = HDdata[Z - 1, N - 1]
    return num / denom if denom != 0 else 0


def prob_P(N_C, N_P, i_C, i_P, Z, N, HDdata):
    if N_C > i_C or N_P > (i_P - 1) or (N - N_C - N_P - 1) > (Z - i_C - i_P):
        return 0
    num = HDdata[i_C, N_C] * HDdata[i_P - 1, N_P] * HDdata[Z - i_C - i_P, N - N_C - N_P - 1]
    denom = HDdata[Z - 1, N - 1]
    return num / denom if denom != 0 else 0


def prob_C(N_C, N_P, i_C, i_P, Z, N, HDdata):
    if N_C > (i_C - 1) or N_P > i_P or (N - N_C - N_P - 1) > (Z - i_C - i_P):
        return 0
    num = HDdata[i_C - 1, N_C] * HDdata[i_P, N_P] * HDdata[Z - i_C - i_P, N - N_C - N_P - 1]
    denom = HDdata[Z - 1, N - 1]
    return num / denom if denom != 0 else 0



def calculate_neighbor_stats(i_C, i_P):
    """预计算邻居统计信息"""
    stats = []
    for N_C in range(0, N):
        for N_P in range(0, N - N_C):
            N_D = N - 1 - N_C - N_P
            if N_D < 0:
                continue


            theta1 = 1 if (N_C + 1) >= M else 0
            theta2 = 1 if N_C >= M else 0
            theta3 = 1 if N_P >= T else 0
            theta4 = 1 if (N_C + 1) >= M1 else 0
            theta5 = 1 if N_C >= M1 else 0


            prob_C_val = prob_C(N_C, N_P, i_C, i_P, Z, N, HDdata)
            prob_D_val = prob_D(N_C, N_P, i_C, i_P, Z, N, HDdata)
            prob_P_val = prob_P(N_C, N_P, i_C, i_P, Z, N, HDdata)

            stats.append((N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5, prob_C_val, prob_D_val, prob_P_val))
    return stats



def pi_C(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
    return ((-c + b * theta1 + (1 - r) * b * (1 - theta1)) * (s - 1) +
            (theta1 * (-c + b * theta4 + (1 - r) * (1 - theta4) * b)+ (1 - theta1) *
             (-c + b * theta1 + (1 - r) * b * (1-theta1))) * (F - s + 1))


def pi_D(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
    return ((b * theta2 + (1 - r) * b * (1 - theta2) - (a1 * theta3)) * (s - 1) +
            (theta2 * (b * theta5 + (1 -r) * b * (1-theta5) - a1 * theta3) + (1 - theta2) *
             (b * theta2 + (1 - r) * b * (1 - theta2) - a2 * theta3)) * (F - s + 1))


def pi_P(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
    return ((-cp + b * theta2 + (1 - r) * b * (1 - theta2)) * (s - 1) +
            (theta2 * (-cp + b * theta5 + (1 - r) * b * (1-theta5)) + (1 - theta2) *
             (-cp + b * theta2 + (1 - r) * b *(1 - theta2))) * (F - s +1))



def fitness_C(i_C, i_P):
    """C类适应度计算"""
    total = 0
    neighbor_stats = calculate_neighbor_stats(i_C, i_P)
    for stat in neighbor_stats:
        N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5, prob_C_val, _, _ = stat
        total += prob_C_val * pi_C(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5)
    return total

def fitness_D(i_C, i_P):
    """D类适应度计算"""
    total = 0
    neighbor_stats = calculate_neighbor_stats(i_C, i_P)
    for stat in neighbor_stats:
        N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5,  _, prob_D_val, _ = stat
        total += prob_D_val * pi_D(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5)
    return total

def fitness_P(i_C, i_P):
    """P类适应度计算"""
    total = 0
    neighbor_stats = calculate_neighbor_stats(i_C, i_P)
    for stat in neighbor_stats:
        N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5, _, _, prob_P_val = stat
        total += prob_P_val * pi_P(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5)
    return total



def Tcd(i_C, i_P, FC, FD):
    return ((1 - mu) * (i_C / Z) * ((Z - i_C - i_P) / (Z - 1)) * (1 / (1 + np.exp(-beta * (FD - FC))))
            + mu * i_C / (2 * Z))


def Tcp(i_C, i_P, FC, FP):
    return ((1 - mu) * (i_C / Z) * (i_P / (Z - 1)) * (1 / (1 + np.exp(-beta * (FP - FC))))
            + mu * i_C / (2 * Z))


def Tdc(i_C, i_P, FD, FC):
    return ((1 - mu) * ((Z - i_C - i_P) / Z) * (i_C / (Z - 1)) * (1 / (1 + np.exp(-beta * (FC - FD))))
            + mu * (Z - i_C - i_P) / (2 * Z))


def Tdp(i_C, i_P, FD, FP):
    return ((1 - mu) * ((Z - i_C - i_P) / Z) * (i_P / (Z - 1)) * (1 / (1 + np.exp(-beta * (FP - FD))))
            + mu * (Z - i_C - i_P) / (2 * Z))


def Tpc(i_C, i_P, FP, FC):
    return ((1 - mu) * (i_P / Z) * (i_C / (Z - 1)) * (1 / (1 + np.exp(-beta * (FC - FP))))
            + mu * i_P / (2 * Z))


def Tpd(i_C, i_P, FP, FD):
    return ((1 - mu) * (i_P / Z) * ((Z - i_C - i_P) / (Z - 1)) * (1 / (1 + np.exp(-beta * (FD - FP))))
            + mu * i_P / (2 * Z))



side = (Z + 1) * (Z + 2) // 2
state_space = np.zeros((side, 2), dtype=int)
index_count = 0
for i_C in range(0, Z + 1):
    for i_P in range(0, Z + 1 - i_C):
        state_space[index_count, :] = [i_C, i_P]
        index_count += 1



def calculate_transfer(index):
    i_C, i_P = state_space[index]


    FC_val = fitness_C(i_C, i_P)
    FD_val = fitness_D(i_C, i_P)
    FP_val = fitness_P(i_C, i_P)


    return [
        Tpc(i_C, i_P, FP_val, FC_val),
        Tcp(i_C, i_P, FC_val, FP_val),
        Tpd(i_C, i_P, FP_val, FD_val),
        Tdp(i_C, i_P, FD_val, FP_val),
        Tdc(i_C, i_P, FD_val, FC_val),
        Tcd(i_C, i_P, FC_val, FD_val),
    ]


zhuan_yi = np.zeros((side, 6))

results = Parallel(n_jobs=-1)(
    delayed(calculate_transfer)(i)
    for i in range(len(state_space))
)
zhuan_yi = np.array(results)


p_tran = np.zeros((side, side))


state_to_index = {}
for idx in range(side):
    i_C, i_P = state_space[idx]
    state_to_index[(i_C, i_P)] = idx


for idx in range(side):
    i_C, i_P = state_space[idx]


    transfers = [
        (i_C - 1, i_P + 1),
        (i_C - 1, i_P),
        (i_C + 1, i_P - 1),
        (i_C, i_P - 1),
        (i_C + 1, i_P),
        (i_C, i_P + 1),
    ]


    probs = [
        zhuan_yi[idx, 1],
        zhuan_yi[idx, 5],
        zhuan_yi[idx, 0],
        zhuan_yi[idx, 2],
        zhuan_yi[idx, 4],
        zhuan_yi[idx, 3],
    ]


    for new_state, prob in zip(transfers, probs):
        if new_state in state_to_index:
            new_idx = state_to_index[new_state]
            p_tran[idx, new_idx] = prob


    p_tran[idx, idx] = 1 - np.sum(probs)


vectors1 = np.zeros((side, 2))
for i in range(side):

    D_up_1 = zhuan_yi[i, 5]
    D_up_2 = zhuan_yi[i, 2]
    D_down_1 = zhuan_yi[i, 4]
    D_down_2 = zhuan_yi[i, 3]
    C_up_2 = zhuan_yi[i, 0]
    C_down_2 = zhuan_yi[i, 1]


    TC = (D_down_1 + C_up_2) - (D_up_1 + C_down_2)
    TD = (D_up_1 + D_up_2) - (D_down_1 + D_down_2)
    TP = (C_down_2 + D_down_2) - (C_up_2 + D_up_2)


    vectors1[i, 0] = TC
    vectors1[i, 1] = TP


vectors2 = np.zeros((side, 2))
for ss in range(side):

    vectors2[ss, 0] = 0.5* vectors1[ss, 0] + vectors1[ss, 1]
    vectors2[ss, 1] = 0.5 * np.sqrt(3) * vectors1[ss, 0]


vector_magnitudes = np.linalg.norm(vectors2, axis=1)
min_mag = np.min(vector_magnitudes)
max_mag = np.max(vector_magnitudes)
if max_mag > min_mag:
    normalized_magnitudes = (vector_magnitudes - min_mag) / (max_mag - min_mag)
else:
    normalized_magnitudes = np.zeros_like(vector_magnitudes)


values, dis_tr = eigs(p_tran.T, k=1, which='LM')
v = dis_tr[:, 0]
v = v.real
v = np.abs(v)
vq = v / np.sum(v)


C_average = 0
P_average = 0
for ii in range(len(vq)):
    i_C, i_P = state_space[ii]

    C_average += vq[ii] * i_C / Z
    P_average += vq[ii] * i_P / Z

D_average = 1 - C_average - P_average

print(f"Average cooperator proportion: {C_average:.4f}")
print(f"Average punisher proportion: {P_average:.4f}")
print(f"Average defector proportion: {D_average:.4f}")


v_normalized = (v - np.min(v)) / (np.max(v) - np.min(v))


total_people = Z
points = []
x_positions = []
y_positions = []
colors = []
u_vectors = []
v_vectors = []


index = 0
for i_C in range(0, Z + 1):
    for i_P in range(0, Z + 1 - i_C):

        x = i_P + i_C / 2.0
        y = np.sqrt(3) * i_C / 2.0


        state = (i_C, i_P)
        if state in state_to_index:
            idx = state_to_index[state]
            if v_normalized[idx]>1:
                color = 1
            else:
                color = v_normalized[idx]
        else:
            color = 0


        points.append((x, y, color))
        x_positions.append(x)
        y_positions.append(y)
        colors.append(color)


        if state in state_to_index:
            idx = state_to_index[state]
            u_vectors.append(vectors2[idx, 0])
            v_vectors.append(vectors2[idx, 1])
        else:
            u_vectors.append(0)
            v_vectors.append(0)


x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
u_vectors = np.array(u_vectors)
v_vectors = np.array(v_vectors)


x_min, x_max = x_positions.min(), x_positions.max()
y_min, y_max = y_positions.min(), y_positions.max()


x_grid, y_grid = np.meshgrid(
    np.linspace(x_min, x_max, 1000),
    np.linspace(y_min, y_max, 1000)
)


u_grid = griddata((x_positions, y_positions), u_vectors, (x_grid, y_grid), method='linear', fill_value=0)
v_grid = griddata((x_positions, y_positions), v_vectors, (x_grid, y_grid), method='linear', fill_value=0)


speed = np.sqrt(u_grid ** 2 + v_grid ** 2)
speed_max = speed.max()
speed_normalized = speed / speed_max


cmap_distribution = colormaps['gist_yarg']
cmap_vectors = colormaps['viridis']


fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal')
norm = Normalize(vmin=0, vmax=1)


scatter = ax.scatter(
    x_positions, y_positions,
    c=colors,
    cmap=cmap_distribution,
    s=2,
    norm=norm,
)


stream = ax.streamplot(
    x_grid, y_grid, u_grid, v_grid,
    color=speed_normalized,
    cmap=cmap_vectors,
    linewidth=1.5,
    density=1.0,
    arrowsize=1.5,
    zorder=2,
    broken_streamlines=True
)


eps = 1.5
triangle_vertices = [
    (0 - eps, 0 - 0.8),
    (Z+ eps, -0.8),
    (Z / 2, np.sqrt(3) * Z / 2+ eps)
]


triangle = plt.Polygon(triangle_vertices, edgecolor='black', fill=None, linewidth=1, zorder=3)
ax.add_patch(triangle)


ax.text(-3, -5, "D", ha='center', fontsize=12)
ax.text(Z + 3, -5, "P", ha='center', fontsize=12)
ax.text(Z / 2, np.sqrt(3) * Z / 2 + 3, "C", ha='center', fontsize=12)


ax.set_xlim(-5, Z + 5)
ax.set_ylim(-5, np.sqrt(3) * (Z + 1) / 2.0 + 5)
ax.axis('off')


divider = make_axes_locatable(ax)
cbar_ax1 = divider.append_axes("right", size="5%", pad=0.05)
cbar_ax2 = divider.append_axes("right", size="5%", pad=0.5)


cbar1 = plt.colorbar(scatter, cax=cbar_ax1)
cbar1.set_label('Stationary Distribution', fontsize=10)

cbar2 = plt.colorbar(stream.lines, cax=cbar_ax2)
cbar2.set_label('Flow Speed', fontsize=10)

plt.tight_layout()
plt.show()