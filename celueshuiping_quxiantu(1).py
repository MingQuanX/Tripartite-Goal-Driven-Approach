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
import pandas as pd


Z = 100
N = 6
M = 2
M1 = 3
T = 1
c = 0.1
b = 1
r = 0.5
a1 = 0.12
#a2 = 0.3
cp = 0.06
mu = 0.01
omega = 0.8
s = 2
F = 1 / (1 - omega)
beta = 5


a2_values = np.arange(0.12, 0.30, 0.01)
#M1_values = [2, 3, 4, 5]

results_list = []


results_df = pd.DataFrame(columns=['a2', 'C_proportion', 'P_proportion', 'D_proportion'])


for a2 in a2_values:
    print(f"\n===== Computing for a2 = {a2} =====")




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

                stats.append(
                    (N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5, prob_C_val, prob_D_val, prob_P_val))
        return stats



    def pi_C(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
        return (-c + b * theta1 + (1 - r) * b * (1 - theta1)) * (s - 1) + (
                theta1 * (-c + b * theta4 + (1 - r) * (1 - theta4) * b) + (1 - theta1) * (
                -c + b * theta1 + (1 - r) * b * (1 - theta1))) * (F - s + 1)


    def pi_D(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
        return (b * theta2 + (1 - r) * b * (1 - theta2) - (a1 * theta3)) * (s - 1) + (
                theta2 * (b * theta5 + (1 - r) * b * (1 - theta5) - a1 * theta3) + (1 - theta2) * (
                b * theta2 + (1 - r) * b * (1 - theta2) - a2 * theta3)) * (F - s + 1)


    def pi_P(N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5):
        return (-cp + b * theta2 + (1 - r) * b * (1 - theta2)) * (s - 1) + (
                theta2 * (-cp + b * theta5 + (1 - r) * b * (1 - theta5)) + (1 - theta2) * (
                -cp + b * theta2 + (1 - r) * b * (1 - theta2))) * (F - s + 1)



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
            N_C, N_P, N_D, theta1, theta2, theta3, theta4, theta5, _, prob_D_val, _ = stat
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



    transition_probs = np.zeros((side, 6))

    results = Parallel(n_jobs=-1)(
        delayed(calculate_transfer)(i)
        for i in range(len(state_space))
    )
    transition_probs = np.array(results)


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
            transition_probs[idx, 1],
            transition_probs[idx, 5],
            transition_probs[idx, 0],
            transition_probs[idx, 2],
            transition_probs[idx, 4],
            transition_probs[idx, 3],
        ]


        for new_state, prob in zip(transfers, probs):
            if new_state in state_to_index:
                new_idx = state_to_index[new_state]
                p_tran[idx, new_idx] = prob


        p_tran[idx, idx] = 1 - np.sum(probs)


    values, dis_tr = eigs(p_tran.T, k=1, which='LM')
    v = dis_tr[:, 0]
    v = v.real
    v = np.abs(v)
    v = v / np.sum(v)


    C_average = 0
    P_average = 0
    for ii in range(len(v)):
        i_C, i_P = state_space[ii]

        C_average += v[ii] * i_C / Z
        P_average += v[ii] * i_P / Z

    D_average = 1 - C_average - P_average

    print(f"a2 = {a2}:")
    print(f"  Average cooperator proportion: {C_average:.4f}")
    print(f"  Average punisher proportion: {P_average:.4f}")
    print(f"  Average defector proportion: {D_average:.4f}")


    results_list.append({
        'a2': a2,
        'C_proportion': C_average,
        'P_proportion': P_average,
        'D_proportion': D_average
    })


results_df = pd.DataFrame(results_list)


print("\n===== Summary of All Results =====")
print(results_df)



plt.figure(figsize=(6, 4.5))
plt.plot(results_df['a2'], results_df['C_proportion'], 'o-',
         color='#1f77b4', label='rate of cooperators', linewidth=2)
plt.plot(results_df['a2'], results_df['P_proportion'], 's-',
         color='#ff7f0e', label='rate of punishers', linewidth=2)
plt.plot(results_df['a2'], results_df['D_proportion'], '^-',
         color='#2ca02c',label='rate of defectors', linewidth=2)

plt.xlabel('a2 (Intensity of punishment)', fontsize=14)
plt.ylabel('rate', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.xlim(0.12, 0.3)


#plt.xticks(results_df['a2'].astype(int))


#for i, row in results_df.iterrows():
#    plt.text(row['a2'], row['C_proportion']+0.01, f"{row['C_proportion']:.3f}",
#             ha='center', va='bottom', fontsize=9)
#    plt.text(row['a2'], row['P_proportion']+0.01, f"{row['P_proportion']:.3f}",
#             ha='center', va='bottom', fontsize=9)
#    plt.text(row['a2'], row['D_proportion']+0.01, f"{row['D_proportion']:.3f}",
#             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()




#plt.figure(figsize=(6, 4.5))


#bar_width = 0.25
#x_pos = np.arange(len(M1_values))


#plt.bar(x_pos - bar_width, results_df['C_proportion'], bar_width,
#        label='rate of cooperators', alpha=0.8, color='#1f77b4')
#plt.bar(x_pos, results_df['P_proportion'], bar_width,
#        label='rate of punishers', alpha=0.8, color='#ff7f0e')
#plt.bar(x_pos + bar_width, results_df['D_proportion'], bar_width,
#        label='rate of defectors', alpha=0.8, color='#2ca02c')

#plt.xlabel('M1 ', fontsize=14)
#plt.ylabel('strategy rate', fontsize=14)
#plt.title('', fontsize=16)
#plt.legend(fontsize=12)


#plt.xticks(x_pos, M1_values)

#plt.ylim(0, 1)

#plt.grid(True, alpha=0.3, axis='y')


#for i, (c_val, p_val, d_val) in enumerate(zip(results_df['C_proportion'],
#                                            results_df['P_proportion'],
#                                            results_df['D_proportion'])):
#    plt.text(i - bar_width, c_val + 0.01, f"{c_val:.3f}",
#             ha='center', va='bottom', fontsize=9)
#    plt.text(i, p_val + 0.01, f"{p_val:.3f}",
#             ha='center', va='bottom', fontsize=9)
#    plt.text(i + bar_width, d_val + 0.01, f"{d_val:.3f}",
#             ha='center', va='bottom', fontsize=9)

#plt.tight_layout()
#plt.show()