def corsort_interruptible(X, max_comparisons):
    """
    Implementação do Corsort que pode ser interrompida após um número máximo de comparações.
    Retorna a melhor estimativa da lista ordenada até o ponto de interrupção.
    """
    start_time = time.perf_counter()  # Inicia a contagem do tempo

    X = list(X)
    n = len(X)
    M = np.zeros((n, n), dtype=int)
    comparisons = []
    estimates = []

    def update_partial_order(i, j):
        """Atualiza a matriz de ordem parcial com base em uma nova comparação."""
        for k in range(n):
            if M[k, i] == 1:
                for l in range(n):
                    if M[j, l] == 1:
                        M[k, l] = 1
                        M[l, k] = -1

    def compute_rho():
        """Calcula o valor rho para cada elemento, usado para estimar a ordem."""
        a = np.sum(M == 1, axis=1)
        d = np.sum(M == 1, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = d / (a + d)
            rho[np.isnan(rho)] = 0.5
        return rho

    rho = compute_rho()

    interrupted = False
    while np.any(M == 0):
        incomparables = [(i, j) for i in range(n) for j in range(i+1, n)
                         if M[i, j] == 0 and M[j, i] == 0]

        if not incomparables:
            break # Nao ha mais comparacoes possiveis

        if len(comparisons) >= max_comparisons:
            interrupted = True
            print(f"--- Interrupção: Alcançado o limite de {max_comparacoes} comparações. ---")
            break

        a = np.sum(M == 1, axis=1)
        d = np.sum(M == 1, axis=0)
        delta = d - a
        I = a + d

        # Tratamento para evitar erros se houver apenas um par incomparavel
        if len(incomparables) == 1:
            best_pair = incomparables[0]
        else:
             # Prioriza pares com a menor diferenca absoluta em delta, depois o maior I
            best_pair = min(incomparables, key=lambda pair: (abs(delta[pair[0]] - delta[pair[1]]),
                                                             max(I[pair[0]], I[pair[1]])))
        i, j = best_pair

        # Realiza a comparação
        if X[i] > X[j]: # Compare X[i] and X[j] directly
            # If X[i] > X[j], swap indices so i refers to the smaller element
            i, j = j, i

        # Now i is the index of the smaller element and j is the index of the larger element
        M[i, j] = 1
        M[j, i] = -1

        comparisons.append((i, j))
        update_partial_order(i, j)
        rho = compute_rho()

        # Gera as estimativas - opcionalmente pode gerar a cada k comparacoes para performance
        sorted_indices = sorted(range(n), key=lambda i: rho[i])
        estimate = [X[i] for i in sorted_indices]
        estimates.append(estimate)

    end_time = time.perf_counter() # Finaliza a contagem do tempo
    print(f"--- Tempo de execução: {end_time - start_time:.6f} segundos. ---")

    return estimates[-1] if estimates else list(X)
