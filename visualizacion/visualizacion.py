# =============================================================================
# MÓDULO: Visualización — Orquestador de Gráficas
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas de salida
DIR_GRAFICAS_EDA     = os.path.join(os.path.dirname(__file__), 'graficas', 'eda')
DIR_GRAFICAS_MODELOS = os.path.join(os.path.dirname(__file__), 'graficas', 'modelos')


# -----------------------------------------------------------------------------
# Gráficas EDA (llama a eda.py)
# -----------------------------------------------------------------------------

def generar_graficas_eda(df, target_col='target'):
    """
    Genera y guarda todas las gráficas del EDA en visualizacion/graficas/eda/.
    Llama directamente a las funciones del módulo eda.
    """
    # Añadir el módulo eda al path si no está
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raiz not in sys.path:
        sys.path.insert(0, raiz)

    from eda.eda import (graficar_histogramas, graficar_boxplots,
                         graficar_correlacion, graficar_target, graficar_pairplot)

    os.makedirs(DIR_GRAFICAS_EDA, exist_ok=True)
    print("\n[Visualización] Generando gráficas EDA...")

    graficar_histogramas(df, save_path=DIR_GRAFICAS_EDA)
    graficar_boxplots(df, save_path=DIR_GRAFICAS_EDA)
    graficar_correlacion(df, save_path=DIR_GRAFICAS_EDA)
    graficar_target(df, target_col=target_col, save_path=DIR_GRAFICAS_EDA)
    graficar_pairplot(df, target_col=target_col, save_path=DIR_GRAFICAS_EDA)

    print(f"  Gráficas EDA guardadas en: {DIR_GRAFICAS_EDA}")


# -----------------------------------------------------------------------------
# Gráficas de Modelos Supervisados
# -----------------------------------------------------------------------------

def graficar_matriz_confusion(cm_array, clases, nombre_modelo, save_path=None):
    """Gráfico de la matriz de confusión para un modelo de clasificación."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=clases, yticklabels=clases,
                linewidths=0.5, linecolor='gray')
    plt.title(f'Matriz de Confusión — {nombre_modelo}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS,
             f'confusion_{nombre_modelo.lower().replace(" ", "_")}.png')
    plt.show()


def graficar_importancia_features(importancias, nombres_features, nombre_modelo,
                                   top_n=15, save_path=None):
    """Gráfico de importancia de variables (para árboles y random forest)."""
    indices = np.argsort(importancias)[-top_n:]
    plt.figure(figsize=(9, max(5, top_n * 0.4)))
    plt.barh(range(len(indices)),
             importancias[indices],
             color='steelblue', edgecolor='white', alpha=0.85)
    plt.yticks(range(len(indices)), [nombres_features[i] for i in indices], fontsize=9)
    plt.xlabel('Importancia')
    plt.title(f'Importancia de Variables — {nombre_modelo}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS,
             f'importancia_{nombre_modelo.lower().replace(" ", "_")}.png')
    plt.show()


def graficar_curva_roc(fpr_dict, tpr_dict, auc_dict, save_path=None):
    """
    Gráfico de curvas ROC para múltiples modelos.
    fpr_dict, tpr_dict, auc_dict: {nombre_modelo: array}
    """
    plt.figure(figsize=(9, 7))
    colores = sns.color_palette('tab10', len(fpr_dict))

    for (nombre, fpr), color in zip(fpr_dict.items(), colores):
        tpr = tpr_dict[nombre]
        auc = auc_dict.get(nombre, None)
        label = f'{nombre} (AUC={auc:.3f})' if auc else nombre
        plt.plot(fpr, tpr, color=color, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Azar')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
    plt.title('Curvas ROC', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS, 'curvas_roc.png')
    plt.show()


def graficar_prediccion_vs_real(y_real, y_pred, nombre_modelo, save_path=None):
    """Gráfico real vs predicho para regresión."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.5, color='steelblue', edgecolors='k',
                linewidths=0.3, s=40)
    lim_min = min(y_real.min(), y_pred.min())
    lim_max = max(y_real.max(), y_pred.max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2, label='Perfecto')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.title(f'Real vs Predicho — {nombre_modelo}', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS,
             f'real_vs_pred_{nombre_modelo.lower().replace(" ", "_")}.png')
    plt.show()


def graficar_comparacion_modelos(df_metricas, save_path=None):
    """
    Gráfico de barras comparando todas las métricas de todos los modelos.
    df_metricas: DataFrame con columnas de métricas e índice = nombre del modelo.
    """
    metricas = [c for c in df_metricas.columns if c not in ('tipo',)]
    n_metricas = len(metricas)

    fig, axes = plt.subplots(1, n_metricas, figsize=(5 * n_metricas, 6))
    if n_metricas == 1:
        axes = [axes]

    colores = sns.color_palette('tab10', len(df_metricas))

    for ax, metrica in zip(axes, metricas):
        valores = df_metricas[metrica]
        barras = ax.bar(df_metricas.index, valores, color=colores, edgecolor='white', alpha=0.85)
        ax.set_title(metrica, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(1.0, valores.max() * 1.15))
        ax.set_ylabel(metrica)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        for barra, val in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS, 'comparacion_modelos.png')
    plt.show()


def graficar_residuos(y_real, y_pred, nombre_modelo, save_path=None):
    """Gráfico de residuos para modelos de regresión."""
    residuos = np.array(y_real) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuos, alpha=0.5, color='steelblue', s=30)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicho')
    axes[0].set_ylabel('Residuo')
    axes[0].set_title(f'Residuos vs Predicho — {nombre_modelo}')

    axes[1].hist(residuos, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Residuo')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'Distribución de Residuos — {nombre_modelo}')

    plt.tight_layout()
    _guardar(save_path or DIR_GRAFICAS_MODELOS,
             f'residuos_{nombre_modelo.lower().replace(" ", "_")}.png')
    plt.show()


# -----------------------------------------------------------------------------
# Pipeline completo de visualización
# -----------------------------------------------------------------------------

def generar_todas_las_graficas(df_original, df_limpio, resultados_modelos,
                                target_col='target'):
    """
    Genera todas las gráficas del proyecto:
    - EDA sobre el dataset original
    - Resultados de modelos supervisados
    """
    generar_graficas_eda(df_original, target_col)

    os.makedirs(DIR_GRAFICAS_MODELOS, exist_ok=True)
    print("\n[Visualización] Generando gráficas de modelos...")

    df_metricas = resultados_modelos.get('metricas')
    if df_metricas is not None:
        graficar_comparacion_modelos(df_metricas, DIR_GRAFICAS_MODELOS)

    for nombre, info in resultados_modelos.get('detalle', {}).items():
        tipo = info.get('tipo', 'clasificacion')

        if tipo == 'clasificacion':
            if 'matriz_confusion' in info:
                graficar_matriz_confusion(
                    info['matriz_confusion'],
                    info.get('clases', []),
                    nombre, DIR_GRAFICAS_MODELOS
                )

        elif tipo == 'regresion':
            if 'y_pred' in info and 'y_real' in info:
                graficar_prediccion_vs_real(
                    info['y_real'], info['y_pred'], nombre, DIR_GRAFICAS_MODELOS
                )
                graficar_residuos(
                    info['y_real'], info['y_pred'], nombre, DIR_GRAFICAS_MODELOS
                )

        if 'importancias' in info:
            feature_names = info.get('feature_names', [f'f{i}' for i in range(len(info['importancias']))])
            graficar_importancia_features(
                np.array(info['importancias']), feature_names, nombre,
                save_path=DIR_GRAFICAS_MODELOS
            )

    print(f"  Gráficas de modelos guardadas en: {DIR_GRAFICAS_MODELOS}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _guardar(save_path, nombre_archivo):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ruta = os.path.join(save_path, nombre_archivo)
        plt.savefig(ruta, bbox_inches='tight', dpi=150)
        print(f"  Guardado: {ruta}")
