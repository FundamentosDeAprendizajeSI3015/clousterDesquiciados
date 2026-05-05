import numpy as np
from .kernel_metrics import complex_ratio, kernel_aligment, FSM, kernel_polarization
from .weight_linear_combination import weight
from .extremality_order import order_compar


METRICS = {
    "alignment":    kernel_aligment,
    "polarization": kernel_polarization,
    "FSM":          FSM,
    "complex_ratio": complex_ratio,
}

DIRECTIONS = {
    "alignment":    1,
    "polarization": 1,
    "FSM":          1,
    "complex_ratio": -1,
}

_custom_metrics = {
    "alignment": kernel_aligment,
    "FSM":       FSM,
}


def metrics_kernels(KL_train, y_train, metrics=None):
    if metrics is None:
        metrics = METRICS

    num_kernels = KL_train.shape[0]
    num_metrics = len(metrics)
    measures = np.zeros((num_kernels, num_metrics))

    for i in range(num_kernels):
        measures[i] = [metrics[m](KL_train[i], y_train) for m in metrics]

    directions = np.array([DIRECTIONS[m] for m in metrics])
    return measures, directions


class KernelWeights:
    def __init__(self, w_1, w_2):
        self.w_1 = w_1
        self.w_2 = w_2


def kernel_extremaly_weights(KL_train, y_train, metrics=None, n=1):
    if metrics is None:
        metrics = _custom_metrics

    measures, direction = metrics_kernels(KL_train, y_train, metrics)

    order_1 = order_compar(measures, direction)
    w_1 = weight(len(order_1) - order_1, n)

    order_2 = order_compar(measures, -1 * direction)
    w_2 = weight(order_2, n)

    return KernelWeights(w_1, w_2)
