# =============================================================================
# MÓDULO: Métricas para Evaluación de Clustering No Supervisado
# Autores: Camila Martínez y María Alejandra Ocampo
# =============================================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, r2_score
)
from sklearn.preprocessing import LabelEncoder


def calcular_metricas_clasificacion(y_true, y_pred, nombre='', average='weighted'):
    """
    Calcula Accuracy, F1 Score, Precisión y Recall entre dos arreglos de etiquetas.

    Parámetros
    ----------
    y_true  : array-like de etiquetas reales (codificadas numéricamente)
    y_pred  : array-like de etiquetas predichas
    nombre  : etiqueta descriptiva del escenario ('Sin corrección' / 'Con corrección')
    average : estrategia de promediado para métricas multiclase ('weighted' por defecto)

    Retorna
    -------
    dict con nombre, accuracy, f1_score, precision, recall
    """
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average=average, zero_division=0)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=average, zero_division=0)

    return {
        'nombre':    nombre,
        'accuracy':  acc,
        'f1_score':  f1,
        'precision': prec,
        'recall':    rec,
    }


def calcular_r2_clusters(y_continuo, etiquetas_cluster):
    """
    Calcula R² usando como predicción la media del target continuo por cluster.

    Un R² alto indica que los clusters explican bien la varianza del target.

    Parámetros
    ----------
    y_continuo       : array-like con el target continuo (e.g., automatizacion 0-100)
    etiquetas_cluster: array-like de IDs de cluster (enteros)

    Retorna
    -------
    float: coeficiente R²
    """
    y   = np.array(y_continuo, dtype=float)
    etq = np.array(etiquetas_cluster)

    y_pred = np.zeros_like(y)
    for cid in np.unique(etq):
        mask = etq == cid
        y_pred[mask] = y[mask].mean()

    return r2_score(y, y_pred)


def calcular_todas_metricas(y_true_cat, y_continuo, etiquetas_raw, etiquetas_corregidas):
    """
    Evalúa el clustering en dos escenarios:

    * Sin corrección: los IDs de cluster crudos (0, 1, 2...) se mapean
      directamente al espacio de clases usando módulo. Representa el caso
      "naive" sin ningún ajuste de etiquetas.

    * Con corrección: las etiquetas ya mapeadas por votación mayoritaria
      (corregir_etiquetas) se comparan con las clases reales.

    Para ambos escenarios se calcula también R² usando la media del target
    continuo por cluster como predicción.

    Parámetros
    ----------
    y_true_cat          : array-like de etiquetas categóricas reales
                          (e.g., ['baja', 'media', 'alta', ...])
    y_continuo          : array-like con el target continuo (e.g., automatizacion)
    etiquetas_raw       : array-like de IDs de cluster sin corregir (enteros)
    etiquetas_corregidas: array-like de etiquetas de clase asignadas por
                          votación mayoritaria (mismas clases que y_true_cat)

    Retorna
    -------
    dict con claves 'sin_correccion', 'con_correccion' y 'clases'
    """
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true_cat)
    n_clases   = len(le.classes_)

    r2_val = calcular_r2_clusters(y_continuo, etiquetas_raw)

    # Sin corrección: IDs crudos mapeados por módulo al rango de clases
    etq_sin      = np.array(etiquetas_raw) % n_clases
    metricas_sin = calcular_metricas_clasificacion(y_true_enc, etq_sin, 'Sin corrección')
    metricas_sin['r2'] = r2_val

    # Con corrección: etiquetas de clase asignadas por votación mayoritaria
    y_pred_corr  = le.transform(np.array(etiquetas_corregidas))
    metricas_con = calcular_metricas_clasificacion(y_true_enc, y_pred_corr, 'Con corrección')
    metricas_con['r2'] = r2_val

    return {
        'sin_correccion': metricas_sin,
        'con_correccion': metricas_con,
        'clases':         le.classes_.tolist(),
    }


def imprimir_reporte(resultados_metricas):
    """Imprime en consola el reporte comparativo de métricas."""
    print("\n" + "="*60)
    print("11. MÉTRICAS — CLUSTERING NO SUPERVISADO")
    print("="*60)

    for clave in ('sin_correccion', 'con_correccion'):
        m = resultados_metricas[clave]
        print(f"\n  [{m['nombre']}]")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    F1 Score  : {m['f1_score']:.4f}")
        print(f"    Precisión : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    R²        : {m['r2']:.4f}")

    print()
