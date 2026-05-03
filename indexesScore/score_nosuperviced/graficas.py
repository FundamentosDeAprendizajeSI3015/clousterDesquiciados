# =============================================================================
# MÓDULO: Gráficas de Métricas — Clustering No Supervisado
# Autores: Camila Martínez y María Alejandra Ocampo
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NOMBRES_DISPLAY = {
    'accuracy':  'Accuracy',
    'f1_score':  'F1 Score',
    'precision': 'Precisión',
    'recall':    'Recall',
    'r2':        'R²',
}

COLOR_SIN = '#e74c3c'  # rojo  → sin corrección
COLOR_CON = '#2ecc71'  # verde → con corrección


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _guardar(save_path, nombre):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ruta = os.path.join(save_path, nombre)
        plt.savefig(ruta, bbox_inches='tight', dpi=150)
        print(f"  Guardado: {ruta}")


def _vals(resultados, metricas_keys):
    sin = [resultados['sin_correccion'][m] for m in metricas_keys]
    con = [resultados['con_correccion'][m] for m in metricas_keys]
    return sin, con


# -----------------------------------------------------------------------------
# 1. Gráfica de barras comparativa — todas las métricas juntas
# -----------------------------------------------------------------------------

def grafica_comparacion_todas(resultados_metricas, save_path=None):
    """
    Barras agrupadas comparando Accuracy, F1, Precisión, Recall y R²
    entre el escenario sin corrección y con corrección de etiquetas.
    """
    keys   = ['accuracy', 'f1_score', 'precision', 'recall', 'r2']
    labels = [NOMBRES_DISPLAY[k] for k in keys]
    vals_sin, vals_con = _vals(resultados_metricas, keys)

    x     = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width / 2, vals_sin, width, label='Sin corrección',
                color=COLOR_SIN, alpha=0.85, edgecolor='k', linewidth=0.7)
    b2 = ax.bar(x + width / 2, vals_con, width, label='Con corrección',
                color=COLOR_CON, alpha=0.85, edgecolor='k', linewidth=0.7)

    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('Comparación de Métricas — Sin vs Con Corrección de Etiquetas\n'
                 'K-Means Clustering No Supervisado',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Valor de la Métrica', fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _guardar(save_path, 'comparacion_todas_metricas.png')
    plt.show()


# -----------------------------------------------------------------------------
# 2. Gráfica individual por métrica
# -----------------------------------------------------------------------------

def grafica_metrica_individual(resultados_metricas, metrica, save_path=None):
    """
    Barra doble (sin / con corrección) para una única métrica.
    Genera un archivo PNG por cada métrica.
    """
    v_sin = resultados_metricas['sin_correccion'][metrica]
    v_con = resultados_metricas['con_correccion'][metrica]
    nombre = NOMBRES_DISPLAY[metrica]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ['Sin corrección', 'Con corrección'],
        [v_sin, v_con],
        color=[COLOR_SIN, COLOR_CON],
        width=0.5, alpha=0.85, edgecolor='k', linewidth=0.7
    )

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title(f'{nombre} — Sin vs Con Corrección de Etiquetas\n'
                 f'K-Means Clustering No Supervisado',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(nombre, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    patch_sin = mpatches.Patch(color=COLOR_SIN, label='Sin corrección')
    patch_con = mpatches.Patch(color=COLOR_CON, label='Con corrección')
    ax.legend(handles=[patch_sin, patch_con], fontsize=10)

    plt.tight_layout()
    _guardar(save_path, f'{metrica}_comparacion.png')
    plt.show()


# -----------------------------------------------------------------------------
# 3. Gráfica de radar (spider chart)
# -----------------------------------------------------------------------------

def grafica_radar_metricas(resultados_metricas, save_path=None):
    """
    Spider chart mostrando las 5 métricas simultáneamente
    para ambos escenarios (sin / con corrección).
    """
    keys   = ['accuracy', 'f1_score', 'precision', 'recall', 'r2']
    labels = [NOMBRES_DISPLAY[k] for k in keys]
    vals_sin, vals_con = _vals(resultados_metricas, keys)

    N      = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    vals_sin_c = vals_sin + vals_sin[:1]
    vals_con_c = vals_con + vals_con[:1]
    angles_c   = angles   + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles_c, vals_sin_c, 'o-', linewidth=2,
            color=COLOR_SIN, label='Sin corrección')
    ax.fill(angles_c, vals_sin_c, alpha=0.18, color=COLOR_SIN)

    ax.plot(angles_c, vals_con_c, 's-', linewidth=2,
            color=COLOR_CON, label='Con corrección')
    ax.fill(angles_c, vals_con_c, alpha=0.18, color=COLOR_CON)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Radar de Métricas — Sin vs Con Corrección de Etiquetas\n'
                 'K-Means Clustering No Supervisado',
                 fontsize=12, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    _guardar(save_path, 'radar_metricas.png')
    plt.show()


# -----------------------------------------------------------------------------
# 4. Panel de subplots — una fila sin corrección, una con corrección
# -----------------------------------------------------------------------------

def grafica_panel_sin_vs_con(resultados_metricas, save_path=None):
    """
    Panel 2×5: fila superior = sin corrección, fila inferior = con corrección.
    Cada columna es una métrica distinta.
    Facilita la comparación visual directa.
    """
    keys   = ['accuracy', 'f1_score', 'precision', 'recall', 'r2']
    labels = [NOMBRES_DISPLAY[k] for k in keys]

    escenarios = [
        ('sin_correccion', 'Sin Corrección de Etiquetas', COLOR_SIN),
        ('con_correccion', 'Con Corrección de Etiquetas', COLOR_CON),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=False)
    fig.suptitle('Panel Comparativo de Métricas — K-Means No Supervisado',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, (clave, titulo_fila, color) in enumerate(escenarios):
        m = resultados_metricas[clave]
        for col, key in enumerate(keys):
            ax = axes[row][col]
            val = m[key]
            bar = ax.bar([NOMBRES_DISPLAY[key]], [val], color=color,
                         alpha=0.85, edgecolor='k', linewidth=0.7, width=0.5)
            ax.text(0, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.15)
            ax.set_title(NOMBRES_DISPLAY[key], fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', labelbottom=False)

            if col == 0:
                ax.set_ylabel(titulo_fila, fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    _guardar(save_path, 'panel_sin_vs_con.png')
    plt.show()


# -----------------------------------------------------------------------------
# Función principal — genera todas las gráficas
# -----------------------------------------------------------------------------

def generar_todas_graficas(resultados_metricas, save_path=None):
    """
    Genera el conjunto completo de gráficas de métricas para el clustering
    no supervisado y las guarda en save_path.

    Archivos generados:
      - comparacion_todas_metricas.png  (barras agrupadas, todas las métricas)
      - accuracy_comparacion.png
      - f1_score_comparacion.png
      - precision_comparacion.png
      - recall_comparacion.png
      - r2_comparacion.png
      - radar_metricas.png              (spider chart)
      - panel_sin_vs_con.png            (panel 2×5)
    """
    print("\n" + "="*60)
    print("GENERANDO GRÁFICAS DE MÉTRICAS — NO SUPERVISADO")
    print("="*60)

    grafica_comparacion_todas(resultados_metricas, save_path)

    for metrica in ['accuracy', 'f1_score', 'precision', 'recall', 'r2']:
        grafica_metrica_individual(resultados_metricas, metrica, save_path)

    grafica_radar_metricas(resultados_metricas, save_path)
    grafica_panel_sin_vs_con(resultados_metricas, save_path)

    print("\n  Todas las gráficas generadas correctamente.")
