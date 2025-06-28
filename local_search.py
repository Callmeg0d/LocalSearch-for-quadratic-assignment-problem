import numpy as np


def calc_cost(A, B, solution):
    A = np.array(A)
    B = np.array(B)
    P = np.array(solution)
    return np.sum(A * B[P[:, None], P[None, :]])


def local_search(A, B, solution=None, max_iter=1000):
    n = len(A)
    current_solution = solution
    current_cost = calc_cost(A, B, current_solution)

    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        best_cost = current_cost
        best_i, best_j = -1, -1

        for i in range(n):
            for j in range(i + 1, n):
                new_solution = current_solution.copy()
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_cost = calc_cost(A, B, new_solution)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_i, best_j = i, j

        if best_cost < current_cost:
            current_solution[best_i], current_solution[best_j] = current_solution[best_j], current_solution[best_i]
            current_cost = best_cost
            improved = True

        iteration += 1

    return current_solution, current_cost


def guided_local_search(A, B, max_gls_iter=50, lambda_param=0.1):
    n = len(A)
    current_solution = np.random.permutation(n)
    penalties = np.zeros((n, n))

    best_solution = None
    best_cost = float('inf')

    for _ in range(max_gls_iter):
        current_solution, current_cost = local_search(A, B, current_solution)

        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost

        utility = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                utility[i][j] = A[i][j] * B[current_solution[i]][current_solution[j]] / (1 + penalties[i][j])

        max_utility = np.max(utility)
        for i in range(n):
            for j in range(n):
                if utility[i][j] == max_utility:
                    penalties[i][j] += lambda_param

    return best_solution, best_cost