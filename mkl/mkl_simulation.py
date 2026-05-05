import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

from .kernel_metrics import complex_ratio, kernel_aligment, FSM, kernel_polarization
from .extremality_weights import kernel_extremaly_weights
from .weak_polynomial_kernel import create_weak_kernels


def _calculate_kernel_metrics(K, y):
    return [
        kernel_aligment(K, y),
        kernel_polarization(K, y),
        FSM(K, y),
        complex_ratio(K),
    ]


def _train_and_evaluate(KL_train, X_train, y_train):
    result = kernel_extremaly_weights(KL_train, y_train, n=2)

    gram_natural = np.einsum('ijk,i->jk', KL_train, result.w_1)
    natural_metrics = _calculate_kernel_metrics(gram_natural, y_train)

    gram_anti = np.einsum('ijk,i->jk', KL_train, result.w_2)
    anti_metrics = _calculate_kernel_metrics(gram_anti, y_train)

    rbf_metrics  = _calculate_kernel_metrics(rbf_kernel(X_train), y_train)
    poly_metrics = _calculate_kernel_metrics(polynomial_kernel(X_train, degree=3), y_train)

    return natural_metrics, anti_metrics, rbf_metrics, poly_metrics


def iteraciones_simulacion(n_iter, num_kernels, t, X, y):
    """
    Runs n_iter random train/test splits, generates num_kernels weak polynomial
    kernels per split, and returns the mean of 4 metrics x 4 methods.

    Returns
    -------
    mean_results : ndarray of shape (4, 4)
        Rows  → Natural, Anti-Natural, RBF, Polynomial
        Cols  → Alignment, Polarization, FSM, Complex-Ratio
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    all_metrics = np.zeros((n_iter, 4, 4))

    for k in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=k
        )
        KL_train, _ = create_weak_kernels(X_train, X_test, t=t, num_kernels=num_kernels)
        nat, anti, rbf, poly = _train_and_evaluate(KL_train, X_train, y_train)

        all_metrics[k, 0] = nat
        all_metrics[k, 1] = anti
        all_metrics[k, 2] = rbf
        all_metrics[k, 3] = poly

    mean_results = np.mean(all_metrics, axis=0)
    return mean_results
