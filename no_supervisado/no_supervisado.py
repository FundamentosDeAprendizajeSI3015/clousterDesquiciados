# =============================================================================
# MÓDULO: Análisis No Supervisado
# Autores: María Alejandra Ocampo y Camila Martínez
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_DISPONIBLE = True
except ImportError:
    UMAP_DISPONIBLE = False
    warnings.warn("umap-learn no está instalado. Instálalo con: pip install umap-learn")

warnings.filterwarnings('ignore', category=FutureWarning)


# -----------------------------------------------------------------------------
# 9. Método del codo para elegir K
# -----------------------------------------------------------------------------

def metodo_codo(X, k_min=2, k_max=12, random_state=42, save_path=None):
    """
    Calcula la inercia para distintos valores de K y grafica el codo.
    Retorna el diccionario de inercias y el K óptimo sugerido.
    """
    print("\n" + "="*60)
    print("9. ANÁLISIS NO SUPERVISADO — K-MEANS")
    print("="*60)
    print("\n[9.1] Método del Codo...")

    inercias = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inercias[k] = km.inertia_

    # Detectar el codo automáticamente (punto de mayor curvatura)
    ks = list(inercias.keys())
    vals = list(inercias.values())
    diffs2 = np.diff(np.diff(vals))
    k_optimo = ks[np.argmax(diffs2) + 1]

    # Gráfico
    plt.figure(figsize=(9, 5))
    plt.plot(ks, vals, 'o-', color='steelblue', linewidth=2, markersize=6)
    plt.axvline(x=k_optimo, color='red', linestyle='--', alpha=0.7,
                label=f'K sugerido = {k_optimo}')
    plt.title('Método del Codo — K-Means', fontsize=13, fontweight='bold')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inercia (WCSS)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _guardar(save_path, 'codo_kmeans.png')
    plt.show()

    print(f"  K óptimo sugerido por el método del codo: {k_optimo}")
    return inercias, k_optimo


# -----------------------------------------------------------------------------
# Silhouette Score
# -----------------------------------------------------------------------------

def evaluar_silhouette(X, k_min=2, k_max=12, random_state=42, save_path=None):
    """
    Calcula el Silhouette Score para distintos valores de K.
    Retorna el diccionario de scores y el K con mejor score.
    """
    print("\n[9.2] Evaluación Silhouette...")

    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = km.fit_predict(X)
        scores[k] = silhouette_score(X, etiquetas)
        print(f"  K={k:2d}  →  Silhouette Score = {scores[k]:.4f}")

    k_mejor = max(scores, key=scores.get)

    plt.figure(figsize=(9, 5))
    plt.plot(list(scores.keys()), list(scores.values()),
             's-', color='darkorange', linewidth=2, markersize=6)
    plt.axvline(x=k_mejor, color='red', linestyle='--', alpha=0.7,
                label=f'K mejor = {k_mejor} (score={scores[k_mejor]:.4f})')
    plt.title('Silhouette Score por K', fontsize=13, fontweight='bold')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _guardar(save_path, 'silhouette_scores.png')
    plt.show()

    print(f"\n  K con mejor Silhouette Score: {k_mejor} (score={scores[k_mejor]:.4f})")
    return scores, k_mejor


def graficar_silhouette_detalle(X, k, random_state=42, save_path=None):
    """Grafica el diagrama de Silhouette detallado para un K específico."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    etiquetas = km.fit_predict(X)
    score_global = silhouette_score(X, etiquetas)
    coefs = silhouette_samples(X, etiquetas)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    colores = cm.nipy_spectral(np.linspace(0, 1, k))

    for i in range(k):
        coefs_i = np.sort(coefs[etiquetas == i])
        size_i = len(coefs_i)
        y_upper = y_lower + size_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, coefs_i,
                         facecolor=colores[i], edgecolor=colores[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=score_global, color='red', linestyle='--',
               label=f'Score promedio = {score_global:.4f}')
    ax.set_title(f'Diagrama Silhouette — K={k}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Coeficiente Silhouette')
    ax.set_ylabel('Cluster')
    ax.legend()
    plt.tight_layout()
    _guardar(save_path, f'silhouette_detalle_k{k}.png')
    plt.show()


# -----------------------------------------------------------------------------
# Ajuste final del modelo K-Means
# -----------------------------------------------------------------------------

def ajustar_kmeans(X, k, random_state=42):
    """Entrena K-Means con el K elegido y retorna el modelo y las etiquetas."""
    print(f"\n[9.3] Entrenando K-Means con K={k}...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    etiquetas = km.fit_predict(X)
    score = silhouette_score(X, etiquetas)
    print(f"  Silhouette Score final: {score:.4f}")
    print(f"  Inercia: {km.inertia_:.2f}")
    return km, etiquetas


# -----------------------------------------------------------------------------
# Visualización de clusters
# -----------------------------------------------------------------------------

def visualizar_clusters_pca(X, etiquetas, titulo='Clusters K-Means (PCA)',
                             save_path=None, nombre_archivo='clusters_pca.png'):
    """Proyecta a 2D con PCA y colorea por cluster."""
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.var_ratio = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=etiquetas, cmap='tab10', alpha=0.7, s=40, edgecolors='k', linewidths=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.title(titulo, fontsize=13, fontweight='bold')
    plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}% var.)')
    plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}% var.)')
    plt.tight_layout()
    _guardar(save_path, nombre_archivo)
    plt.show()


def visualizar_clusters_umap(X, etiquetas, save_path=None):
    """Proyecta a 2D con UMAP y colorea por cluster."""
    if not UMAP_DISPONIBLE:
        print("  UMAP no disponible, omitiendo visualización UMAP.")
        return

    print("  Calculando proyección UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                          c=etiquetas, cmap='tab10', alpha=0.7, s=40,
                          edgecolors='k', linewidths=0.3)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clusters K-Means — Proyección UMAP', fontsize=13, fontweight='bold')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    _guardar(save_path, 'clusters_umap.png')
    plt.show()


# -----------------------------------------------------------------------------
# 10. Corrección de etiquetas
# -----------------------------------------------------------------------------

def corregir_etiquetas(df, etiquetas_cluster, target_col='target'):
    """
    10. Reasigna las etiquetas del target basándose en los patrones encontrados
    por K-Means. Usa votación mayoritaria por cluster para mapear
    cluster → clase real.
    """
    print("\n" + "="*60)
    print("10. CORRECCIÓN DE ETIQUETAS")
    print("="*60)

    df_temp = df.copy()
    df_temp['_cluster_'] = etiquetas_cluster
    df_temp['_target_'] = df[target_col].values

    # Para cada cluster, encontrar la clase mayoritaria
    mapeo = {}
    for cluster_id in sorted(df_temp['_cluster_'].unique()):
        subset = df_temp[df_temp['_cluster_'] == cluster_id]['_target_']
        clase_mayoritaria = subset.mode()[0]
        mapeo[cluster_id] = clase_mayoritaria
        print(f"  Cluster {cluster_id} → Clase real predominante: {clase_mayoritaria}")

    etiquetas_corregidas = pd.Series(etiquetas_cluster).map(mapeo).values

    # Resumen de concordancia
    coincidencias = (etiquetas_corregidas == df[target_col].values).mean() * 100
    print(f"\n  Concordancia cluster-etiqueta real: {coincidencias:.1f}%")

    return etiquetas_corregidas, mapeo


# -----------------------------------------------------------------------------
# Pipeline No Supervisado
# -----------------------------------------------------------------------------

def ejecutar_no_supervisado(X, df=None, target_col='target',
                             k_min=2, k_max=12, k_forzado=None,
                             random_state=42, save_path=None):
    """
    Pipeline completo:
    1. Método del codo
    2. Silhouette scores
    3. Ajuste K-Means con el K elegido
    4. Visualización de clusters (PCA + UMAP)
    5. Corrección de etiquetas (si se provee df con target)
    """
    # Selección de K
    _, k_codo      = metodo_codo(X, k_min, k_max, random_state, save_path)
    scores_sil, k_sil = evaluar_silhouette(X, k_min, k_max, random_state, save_path)

    k_final = k_forzado if k_forzado else k_sil
    print(f"\n  K seleccionado: {k_final}  (codo={k_codo}, silhouette={k_sil})")

    graficar_silhouette_detalle(X, k_final, random_state, save_path)

    modelo_km, etiquetas = ajustar_kmeans(X, k_final, random_state)

    visualizar_clusters_pca(X, etiquetas, save_path=save_path)
    visualizar_clusters_umap(X, etiquetas, save_path=save_path)

    etiquetas_corregidas = None
    mapeo = None
    if df is not None and target_col in df.columns:
        etiquetas_corregidas, mapeo = corregir_etiquetas(df, etiquetas, target_col)

    resultados = {
        'modelo':               modelo_km,
        'etiquetas':            etiquetas,
        'etiquetas_corregidas': etiquetas_corregidas,
        'mapeo_clusters':       mapeo,
        'k_final':              k_final,
        'silhouette_scores':    scores_sil,
    }
    return resultados


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _guardar(save_path, nombre_archivo):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ruta = os.path.join(save_path, nombre_archivo)
        plt.savefig(ruta, bbox_inches='tight', dpi=150)
        print(f"  Guardado: {ruta}")
