import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def create_weak_kernels(X_train, X_test=None, t=5, num_kernels=3, max_degree=3):
    KL_train = []
    KL_test = None if X_test is None else []

    for _ in range(num_kernels):
        n_cols = np.random.randint(1, t + 1)
        index_columns = np.random.randint(0, X_train.shape[1], size=n_cols)

        X1 = X_train[:, index_columns]
        degree = np.random.randint(1, max_degree + 1)

        KL_train.append(polynomial_kernel(X1, degree=degree, coef0=0, gamma=1))

        if X_test is not None:
            X2 = X_test[:, index_columns]
            KL_test.append(polynomial_kernel(X2, X1, degree=degree, coef0=0, gamma=1))

    KL_train = np.array(KL_train)
    KL_test = np.array(KL_test) if X_test is not None else None

    return (KL_train, KL_test) if X_test is not None else KL_train
