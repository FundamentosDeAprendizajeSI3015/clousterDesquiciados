import os
import numpy as np
import matplotlib.pyplot as plt

from .mkl_simulation import iteraciones_simulacion


_METRIC_LABELS    = ["Kernel Alignment", "Kernel Polarization", "FSM", "Complex Ratio"]
_ALGORITHM_LABELS = ["Natural", "Anti-Natural", "RBF", "Polynomial"]


def plot_metrics_vs_num_kernels(n_iter, t, X, y, kernels_list, save_path=None):
    """
    For each value in kernels_list, runs the MKL simulation and plots how the
    4 kernel metrics evolve as a function of the number of weak kernels.

    Returns a dict mapping num_kernels → mean_results array (4×4).
    """
    print(f"\n  MKL: evaluando kernels_list={kernels_list}, iter={n_iter} ...")
    results = {}
    for nk in kernels_list:
        print(f"    num_kernels={nk} ...", end=" ", flush=True)
        results[nk] = iteraciones_simulacion(n_iter, nk, t, X, y)
        print("listo")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Evolución de métricas de kernel según número de kernels débiles", fontsize=13)

    for idx, ax in enumerate(axes.flat):
        for j, label in enumerate(_ALGORITHM_LABELS):
            vals = [results[nk][j, idx] for nk in kernels_list]
            ax.plot(kernels_list, vals, marker='o', label=label)
        ax.set_title(_METRIC_LABELS[idx])
        ax.set_xlabel("Número de kernels")
        ax.set_ylabel("Valor de la métrica")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, "mkl_metrics_vs_num_kernels.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gráfica guardada: {fig_path}")

    plt.close(fig)
    return results


def plot_metric_heatmap(mean_results, num_kernels, save_path=None):
    """
    Plots a heatmap of the 4×4 mean results matrix for a given num_kernels run.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mean_results, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(_METRIC_LABELS, rotation=20, ha='right', fontsize=9)
    ax.set_yticklabels(_ALGORITHM_LABELS, fontsize=9)
    ax.set_title(f"MKL — métricas promedio (num_kernels={num_kernels})", fontsize=11)
    plt.colorbar(im, ax=ax)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mean_results[i, j]:.3f}", ha='center', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, f"mkl_heatmap_k{num_kernels}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"  Heatmap guardado: {fig_path}")

    plt.close(fig)
