def corsort(X, max_comparisons=None):
    X = list(X)
    n = len(X)
    M = np.zeros((n, n), dtype=int)
    comparisons = []
    estimates = []

    def update_partial_order(i, j):
        nonlocal M
        for k in range(n):
            if M[k, i] == 1:
                for l in range(n):
                    if M[j, l] == 1:
                        M[k, l] = 1
                        M[l, k] = -1

    def compute_a_d_I_delta_rho():
        a = np.sum(M == 1, axis=1)
        d = np.sum(M == 1, axis=0)
        I = a + d
        delta = d - a
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = d / (a + d)
            rho[np.isnan(rho)] = 0.5
        return a, d, I, delta, rho

    a, d, I, delta, rho = compute_a_d_I_delta_rho()

    while np.any(M == 0):
        incomparables = [(i, j) for i in range(n) for j in range(i+1, n)
                         if M[i, j] == 0 and M[j, i] == 0]

        if not incomparables or (max_comparisons and len(comparisons) >= max_comparisons):
            break

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

        a, d, I, delta, rho = compute_a_d_I_delta_rho()
        sorted_indices = sorted(range(n), key=lambda i: rho[i])
        estimate = [X[i] for i in sorted_indices]
        estimates.append(list(estimate))

    return estimates, comparisons

def corsort_batch(n_listas=5, tamanho=100, seed=42):
    np.random.seed(seed)
    tempos = []
    resultados = []

    print('--- Ordenando e comparando o tempo de cada lista ---\n')

    for i in range(n_listas):
        X = np.random.randint(0, 1000, tamanho)
        true_sorted = sorted(X)

        start = time.perf_counter()
        estimates, _ = corsort(X)
        end = time.perf_counter()

        final_estimate = estimates[-1] if estimates else list(X)
        tempo = end - start
        tempos.append(tempo)
        resultados.append((i + 1, final_estimate, tempo))

        print(f"Lista {i+1} ordenada:")
        for j in range(0, len(final_estimate), 30):
          print([int(x) for x in final_estimate[j:j+30]])
        print()

    print()
    for i, _, tempo in resultados:
        print(f"Tempo da Lista {i}: {tempo:.6f} segundos")

    mais_rapida = min(resultados, key=lambda x: x[2])
    print(f"\n**A lista que foi ordenada mais rapidamente é a Lista {mais_rapida[0]} com um tempo de {mais_rapida[2]:.6f} segundos.**")

corsort_batch()
