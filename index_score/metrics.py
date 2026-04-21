# =============================================================================
# MÓDULO: Índices, Scores y Comparación de Modelos
# =============================================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


# =============================================================================
# 11. MODELOS
# =============================================================================

# -----------------------------------------------------------------------------
# Regresión Lineal
# -----------------------------------------------------------------------------

def entrenar_regresion_lineal(X_train, y_train, X_test, y_test):
    """Entrena Regresión Lineal y retorna modelo + métricas."""
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    resultado = {
        'modelo':   modelo,
        'tipo':     'regresion',
        'y_real':   y_test,
        'y_pred':   y_pred_test,
        'metricas': {
            'R2_train': r2_score(y_train, y_pred_train),
            'R2_test':  r2_score(y_test,  y_pred_test),
            'MSE':      mean_squared_error(y_test, y_pred_test),
            'RMSE':     np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE':      mean_absolute_error(y_test, y_pred_test),
        }
    }
    _imprimir_metricas('Regresión Lineal', resultado['metricas'])
    return resultado


# -----------------------------------------------------------------------------
# Regresión Logística
# -----------------------------------------------------------------------------

def entrenar_regresion_logistica(X_train, y_train, X_test, y_test,
                                  max_iter=1000, random_state=42):
    """Entrena Regresión Logística y retorna modelo + métricas de clasificación."""
    modelo = LogisticRegression(max_iter=max_iter, random_state=random_state)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    clases = modelo.classes_

    resultado = {
        'modelo':           modelo,
        'tipo':             'clasificacion',
        'y_real':           y_test,
        'y_pred':           y_pred,
        'clases':           clases,
        'matriz_confusion': confusion_matrix(y_test, y_pred),
        'metricas':         _metricas_clasificacion(y_test, y_pred),
    }
    _imprimir_metricas('Regresión Logística', resultado['metricas'])
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in clases]))
    return resultado


# -----------------------------------------------------------------------------
# Árbol de Decisión
# -----------------------------------------------------------------------------

def entrenar_arbol_decision(X_train, y_train, X_test, y_test,
                             max_depth=None, random_state=42):
    """Entrena un Árbol de Decisión y retorna modelo + métricas."""
    modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    resultado = {
        'modelo':           modelo,
        'tipo':             'clasificacion',
        'y_real':           y_test,
        'y_pred':           y_pred,
        'clases':           modelo.classes_,
        'matriz_confusion': confusion_matrix(y_test, y_pred),
        'importancias':     modelo.feature_importances_,
        'metricas':         _metricas_clasificacion(y_test, y_pred),
    }
    _imprimir_metricas('Árbol de Decisión', resultado['metricas'])
    return resultado


# -----------------------------------------------------------------------------
# Random Forest
# -----------------------------------------------------------------------------

def entrenar_random_forest(X_train, y_train, X_test, y_test,
                            n_estimators=100, max_depth=None, random_state=42):
    """Entrena Random Forest y retorna modelo + métricas."""
    modelo = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    resultado = {
        'modelo':           modelo,
        'tipo':             'clasificacion',
        'y_real':           y_test,
        'y_pred':           y_pred,
        'clases':           modelo.classes_,
        'matriz_confusion': confusion_matrix(y_test, y_pred),
        'importancias':     modelo.feature_importances_,
        'metricas':         _metricas_clasificacion(y_test, y_pred),
    }
    _imprimir_metricas('Random Forest', resultado['metricas'])
    return resultado


# -----------------------------------------------------------------------------
# SVM Clasificación
# -----------------------------------------------------------------------------

def entrenar_svm_clasificacion(X_train, y_train, X_test, y_test,
                                kernel='rbf', C=1.0, random_state=42):
    """Entrena SVM para clasificación y retorna modelo + métricas."""
    modelo = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    resultado = {
        'modelo':           modelo,
        'tipo':             'clasificacion',
        'y_real':           y_test,
        'y_pred':           y_pred,
        'clases':           modelo.classes_,
        'matriz_confusion': confusion_matrix(y_test, y_pred),
        'metricas':         _metricas_clasificacion(y_test, y_pred),
    }
    _imprimir_metricas(f'SVM (kernel={kernel})', resultado['metricas'])
    return resultado


# -----------------------------------------------------------------------------
# SVM Regresión
# -----------------------------------------------------------------------------

def entrenar_svm_regresion(X_train, y_train, X_test, y_test,
                            kernel='rbf', C=1.0):
    """Entrena SVR para regresión y retorna modelo + métricas."""
    modelo = SVR(kernel=kernel, C=C)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    resultado = {
        'modelo':  modelo,
        'tipo':    'regresion',
        'y_real':  y_test,
        'y_pred':  y_pred,
        'metricas': {
            'R2':   r2_score(y_test, y_pred),
            'MSE':  mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE':  mean_absolute_error(y_test, y_pred),
        }
    }
    _imprimir_metricas(f'SVR (kernel={kernel})', resultado['metricas'])
    return resultado


# =============================================================================
# 12. MÉTRICAS Y COMPARACIÓN
# =============================================================================

def comparar_modelos(resultados_dict):
    """
    Genera una tabla comparativa de todos los modelos.
    resultados_dict: {nombre: resultado} donde resultado viene de las funciones anteriores.
    Retorna un DataFrame con las métricas de cada modelo.
    """
    print("\n" + "="*60)
    print("COMPARACIÓN DE MODELOS")
    print("="*60)

    filas = []
    for nombre, res in resultados_dict.items():
        fila = {'Modelo': nombre, 'Tipo': res['tipo']}
        fila.update(res['metricas'])
        filas.append(fila)

    df = pd.DataFrame(filas).set_index('Modelo')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df.round(4).to_string())

    # Mejor modelo por tipo
    clf_modelos = df[df['Tipo'] == 'clasificacion']
    reg_modelos = df[df['Tipo'] == 'regresion']

    if not clf_modelos.empty and 'Accuracy' in clf_modelos.columns:
        mejor_clf = clf_modelos['Accuracy'].idxmax()
        print(f"\n  Mejor modelo de CLASIFICACIÓN (Accuracy): {mejor_clf}"
              f"  →  {clf_modelos.loc[mejor_clf, 'Accuracy']:.4f}")

    if not reg_modelos.empty and 'R2_test' in reg_modelos.columns:
        mejor_reg = reg_modelos['R2_test'].idxmax()
        print(f"  Mejor modelo de REGRESIÓN (R²): {mejor_reg}"
              f"  →  {reg_modelos.loc[mejor_reg, 'R2_test']:.4f}")
    elif not reg_modelos.empty and 'R2' in reg_modelos.columns:
        mejor_reg = reg_modelos['R2'].idxmax()
        print(f"  Mejor modelo de REGRESIÓN (R²): {mejor_reg}"
              f"  →  {reg_modelos.loc[mejor_reg, 'R2']:.4f}")

    return df


def guardar_metricas(df_metricas, ruta='data/metricas_modelos.csv'):
    """Guarda la tabla de métricas en un CSV."""
    import os
    os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else '.', exist_ok=True)
    df_metricas.to_csv(ruta)
    print(f"\n  Métricas guardadas en: {ruta}")


# =============================================================================
# Helpers internos
# =============================================================================

def _metricas_clasificacion(y_real, y_pred):
    promedio = 'weighted'
    return {
        'Accuracy':  accuracy_score(y_real, y_pred),
        'F1':        f1_score(y_real, y_pred, average=promedio, zero_division=0),
        'Precision': precision_score(y_real, y_pred, average=promedio, zero_division=0),
        'Recall':    recall_score(y_real, y_pred, average=promedio, zero_division=0),
    }


def _imprimir_metricas(nombre, metricas):
    print(f"\n  --- {nombre} ---")
    for k, v in metricas.items():
        print(f"    {k:<15} {v:.4f}")
