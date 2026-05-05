import numpy as np


def complex_ratio(A):
    return np.trace(A)


def FSM(K, y):
    n_nega = np.count_nonzero(y == -1)
    n_posi = np.count_nonzero(y == 1)

    if n_nega == 0 or n_posi == 0:
        return 0.0

    d_i = np.sum(K[np.ix_(y == -1, y == -1)], axis=1) / n_nega
    a_i = np.sum(K[np.ix_(y == 1,  y == 1)],  axis=1) / n_posi
    c_i = np.sum(K[np.ix_(y == -1, y == 1)],  axis=1) / n_posi
    b_i = np.sum(K[np.ix_(y == 1,  y == -1)], axis=1) / n_nega

    A = np.sum(a_i) / n_posi
    B = np.sum(b_i) / n_posi
    C = np.sum(c_i) / n_nega
    D = np.sum(d_i) / n_nega

    rest_phi_square = A + D - B - C
    if rest_phi_square == 0:
        return 0.0

    aux_1 = np.sum(np.square(b_i - a_i + A - B)) / (rest_phi_square * max(n_posi - 1, 1))
    aux_2 = np.sum(np.square(c_i - d_i + D - C)) / (rest_phi_square * max(n_nega - 1, 1))

    return (np.sqrt(aux_1) + np.sqrt(aux_2)) / np.sqrt(rest_phi_square)


def ideal_kernel(y):
    K_ideal = np.equal.outer(y, y).astype(int)
    K_ideal = np.where(K_ideal == 0, -1, K_ideal)
    return K_ideal


def kernel_aligment(K, y):
    A1 = np.trace(np.dot(K.T, ideal_kernel(y)))
    norm = np.linalg.norm(K, 'fro') * len(y)
    if norm == 0:
        return 0.0
    return A1 / norm


def kernel_polarization(K, y):
    # Vectorized implementation (avoids O(n²) Python loop)
    diag_k = np.diag(K)
    outer_y = np.outer(y, y)
    D = diag_k[:, None] + diag_k[None, :] - 2 * K
    A = -outer_y * D
    np.fill_diagonal(A, 0)
    return float(np.sum(A))
