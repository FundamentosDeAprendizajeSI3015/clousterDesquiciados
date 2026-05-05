import numpy as np


def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]

    return Q, R


def matrix_rotation(u):
    n = len(u)
    x = np.identity(n)
    M_u = np.multiply(np.reshape(np.sign(u), (len(u), 1)), x)
    M_u[:, 0] = u / np.linalg.norm(u, 2)
    x[:, 0] = np.ones(n) / np.sqrt(n)
    M_e = x

    qu, ru = gram_schmidt(M_u)
    qe, re = gram_schmidt(M_e)

    R_u = np.matmul(qe, np.transpose(qu))
    return R_u


def order_compar(kernels_metrics, u):
    num_kernels = np.shape(kernels_metrics)[0]

    Ru = matrix_rotation(u)
    new_metrics = np.transpose(np.dot(Ru, np.transpose(kernels_metrics)))

    extremality_order = np.zeros(num_kernels)
    for i in range(num_kernels):
        extremality_order[i] = np.sum(np.all(new_metrics >= new_metrics[i], axis=1))

    return extremality_order
