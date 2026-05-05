import numpy as np
import pandas as pd

from .mkl_plots import plot_metrics_vs_num_kernels, plot_metric_heatmap


def _convertir_etiquetas_binarias(df, target_col):
    """
    Converts target_col to binary labels {-1, +1} required by MKL metrics.

    - If exactly 2 unique classes: first class → -1, second → +1.
    - If more than 2 classes: keeps only the two most frequent classes.

    Returns (X_mask, y_bin, clases_usadas).
    """
    valores = df[target_col].values
    clases = sorted(df[target_col].unique())

    if len(clases) == 2:
        y_bin = np.where(valores == clases[0], -1, 1)
        mask = np.ones(len(df), dtype=bool)
        print(f"  Clases binarias detectadas: {clases[0]} → -1 | {clases[1]} → +1")
    else:
        conteos = pd.Series(valores).value_counts()
        top2 = conteos.index[:2].tolist()
        mask = df[target_col].isin(top2).values
        vals_filtrados = valores[mask]
        y_bin = np.where(vals_filtrados == top2[0], -1, 1)
        print(f"  Más de 2 clases en '{target_col}'. Se usan las dos más frecuentes:")
        print(f"    {top2[0]} → -1 | {top2[1]} → +1  ({mask.sum()} muestras)")

    return mask, y_bin, clases[:2]


def ejecutar_mkl(
    df,
    features,
    target_col="automatizacion_cat",
    save_path=None,
    n_iter=10,
    kernels_list=None,
    t=4,
):
    """
    Runs the Extremality Multiple Kernel Learning (EMKL) analysis.

    Parameters
    ----------
    df           : cleaned DataFrame from the pipeline
    features     : list of feature column names
    target_col   : binary classification target (will be binarised if needed)
    save_path    : directory where plots are saved
    n_iter       : number of random train/test iterations per num_kernels value
    kernels_list : list of num_kernels values to evaluate (default [5, 10, 20])
    t            : max number of features per weak kernel (default 4, ≤ len(features))

    Returns
    -------
    dict with keys: 'results' (dict nk→mean_matrix), 'clases', 'n_samples'
    """
    if kernels_list is None:
        kernels_list = [5, 10, 20]

    t = min(t, len(features))

    print("\n" + "=" * 60)
    print("MKL — MULTIPLE KERNEL LEARNING (EXTREMALITY)")
    print("=" * 60)

    mask, y, clases = _convertir_etiquetas_binarias(df, target_col)
    X = df[features].values[mask]

    print(f"  Muestras: {len(y)} | Features: {len(features)} | t={t}")
    print(f"  Clases → -1: {(y==-1).sum()}  |  +1: {(y==1).sum()}")
    print(f"  Iteraciones por configuración: {n_iter}")
    print(f"  Configuraciones de kernels: {kernels_list}")

    results = plot_metrics_vs_num_kernels(
        n_iter=n_iter,
        t=t,
        X=X,
        y=y,
        kernels_list=kernels_list,
        save_path=save_path,
    )

    best_nk = kernels_list[-1]
    plot_metric_heatmap(results[best_nk], num_kernels=best_nk, save_path=save_path)

    _imprimir_resumen(results, kernels_list)

    return {
        "results":   results,
        "clases":    clases,
        "n_samples": int(len(y)),
    }


def _imprimir_resumen(results, kernels_list):
    metric_names = ["Alignment", "Polarization", "FSM", "Complex-Ratio"]
    method_names = ["Natural", "Anti-Natural", "RBF", "Polynomial"]

    print("\n" + "=" * 60)
    print("RESUMEN MKL")
    print("=" * 60)

    for nk in kernels_list:
        mat = results[nk]
        print(f"\n  num_kernels = {nk}")
        header = f"  {'Método':<14}" + "".join(f"{m:>14}" for m in metric_names)
        print(header)
        print("  " + "-" * (14 + 14 * 4))
        for i, method in enumerate(method_names):
            row = f"  {method:<14}" + "".join(f"{mat[i, j]:>14.4f}" for j in range(4))
            print(row)
