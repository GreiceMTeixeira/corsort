import numpy as np
import math
import random
from scipy.stats import rankdata
import matplotlib.pyplot as plt

def spearman_footrule(estimate, true_sorted):
    rank_estimate = rankdata(estimate, method='ordinal')
    rank_true = rankdata(true_sorted, method='ordinal')
    return sum(abs(rank_estimate[i] - rank_true[i]) for i in range(len(estimate)))

def corsort_interruptible(X, max_comparisons):
    X = list(X)
    n = len(X)
    M = np.zeros((n, n), dtype=int)
    comparisons = []

    def update_partial_order(i, j):
        for k in range(n):
            if M[k, i] == 1:
                for l in range(n):
                    if M[j, l] == 1:
                        M[k, l] = 1
                        M[l, k] = -1

    def compute_rho():
        a = np.sum(M == 1, axis=1)
        d = np.sum(M == 1, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = d / (a + d)
            rho[np.isnan(rho)] = 0.5
        return rho

    rho = compute_rho()

    while np.any(M == 0):
        incomparables = [(i, j) for i in range(n) for j in range(i+1, n)
                         if M[i, j] == 0 and M[j, i] == 0]
        if not incomparables or len(comparisons) >= max_comparisons:
            break

        a = np.sum(M == 1, axis=1)
        d = np.sum(M == 1, axis=0)
        delta = d - a
        I = a + d

        best_pair = min(incomparables, key=lambda pair: (abs(delta[pair[0]] - delta[pair[1]]),
                                                         max(I[pair[0]], I[pair[1]])))
        i, j = best_pair

        if X[i] < X[j]:
            M[i, j] = 1
            M[j, i] = -1
        else:
            M[i, j] = -1
            M[j, i] = 1
            i, j = j, i

        comparisons.append((i, j))
        update_partial_order(i, j)
        rho = compute_rho()

    sorted_indices = sorted(range(n), key=lambda i: rho[i])
    return [X[i] for i in sorted_indices]

def multizip_sort_interruptible(x, max_comparisons):
    comparison_counter = [0]
    y_lists = [x]

    def merge_interrupt(lista1, lista2):
        result = []
        ptr1 = ptr2 = 0
        while ptr1 < len(lista1) and ptr2 < len(lista2):
            if comparison_counter[0] >= max_comparisons:
                break
            comparison_counter[0] += 1
            if lista1[ptr1] <= lista2[ptr2]:
                result.append(lista1[ptr1])
                ptr1 += 1
            else:
                result.append(lista2[ptr2])
                ptr2 += 1
        result.extend(lista1[ptr1:])
        result.extend(lista2[ptr2:])
        return result

    while any(len(lst) > 1 for lst in y_lists):
        novas = []
        for y in y_lists:
            meio = math.ceil(len(y) / 2)
            novas.append(y[:meio])
            novas.append(y[meio:])
        y_lists = novas

    while len(y_lists) > 1:
        nova_y = []
        i = 0
        while i + 1 < len(y_lists):
            merged = merge_interrupt(y_lists[i], y_lists[i + 1])
            nova_y.append(merged)
            i += 2
        if i < len(y_lists):
            nova_y.append(y_lists[i])
        y_lists = nova_y
        if comparison_counter[0] >= max_comparisons:
            break

    resultado_final = []
    for sub in y_lists:
        resultado_final.extend(sub)
    return resultado_final

# Comparação visual
random.seed(42)
X = [random.randint(0, 1000) for _ in range(100)]
true_sorted = sorted(X)

ks = list(range(10, 101, 10))
errors_corsort = []
errors_multizip = []

for k in ks:
    est_corsort = corsort_interruptible(X, k)
    est_multizip = multizip_sort_interruptible(X, k)
    errors_corsort.append(spearman_footrule(est_corsort, true_sorted))
    errors_multizip.append(spearman_footrule(est_multizip, true_sorted))

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(ks, errors_corsort, marker='o', label='Corsort (interrompido)')
plt.plot(ks, errors_multizip, marker='s', label='Multizip (interrompido)')
plt.xlabel('Número de comparações permitidas')
plt.ylabel('Erro de ordenação (Spearman Footrule)')
plt.title('Comparação Anytime: Corsort vs Multizip')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
