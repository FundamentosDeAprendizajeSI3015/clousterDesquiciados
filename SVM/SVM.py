# Autores: Luciana Hoyos Pérez y Santiago Manco Maya

"""
Máquinas de Soporte Vectorial (SVM) — Luciana Hoyos Pérez y Santiago Manco Maya
Ejecuta 4 kernels SVM sobre el dataset de reemplazabilidad IA:
  · 1 Lineal
  · 2 No lineales (Polinómico y Sigmoide)
  · 1 Radial (RBF)

Las gráficas se guardan automáticamente en visualizaciones/SVM/.

Uso desde la raíz del proyecto:
    python SVM/SVM.py
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ── Rutas ─────────────────────────────────────────────────────────────────────

# Raíz del proyecto: un nivel arriba de este archivo
ROOT_DIR   = Path(__file__).resolve().parent.parent
PLOTS_DIR  = ROOT_DIR / "visualizaciones" / "SVM"
DATA_PATH  = ROOT_DIR / "dataset_reemplazabilidad_ia.csv"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
DPI          = 150

FEATURES = [
    "nivel_experiencia",
    "nivel_rutina",
    "nivel_estructuracion",
    "nivel_creatividad",
    "resolucion_problemas_complejos",
    "interaccion_humana",
]
TARGET = "porcentaje_tareas_automatizables"


# ── Estructura de datos ────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """Contiene todas las versiones del dataset necesarias para los modelos SVM."""
    X_train_sc: np.ndarray   # Escalado — SVM es sensible a la magnitud
    X_test_sc:  np.ndarray
    y_train:    np.ndarray
    y_test:     np.ndarray
    feature_names: list
    classes:    np.ndarray   # Clases únicas del target
    # Dataset completo (para learning_curve con CV)
    X_full_sc:  np.ndarray
    y_full:     np.ndarray
    label_encoder: LabelEncoder


# ── Helpers ────────────────────────────────────────────────────────────────────

def _preparar_carpeta_visualizaciones() -> None:
    """
    Crea visualizaciones/SVM/ si no existe.
    El patrón os.makedirs(..., exist_ok=True) evita error si ya existe.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Carpeta de visualizaciones lista: {PLOTS_DIR}")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Guarda y cierra la figura. Centraliza el patrón savefig/close/print."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Gráfica guardada → {path}")


def _categorizar_automatizacion(x: float) -> str:
    """
    Convierte el porcentaje continuo de automatización en 3 categorías:
      · baja  : 0–1  (poco automatizable)
      · media : 2–3  (automatización moderada)
      · alta  : 4–6  (muy automatizable)
    Replica la lógica usada en Limpieza.py.
    """
    if x < 2:
        return "baja"
    elif x < 4:
        return "media"
    else:
        return "alta"


def _calc_metricas_clasificacion(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Calcula accuracy, precision, recall, F1 y AUC-ROC.
    Usa promedio 'macro' para dar igual peso a todas las clases,
    lo cual es más justo ante desbalance de clases.
    """
    multiclase = len(np.unique(y_test)) > 2
    avg = "macro" if multiclase else "binary"

    metricas = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
    }

    if multiclase:
        metricas["AUC-ROC"] = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )
    else:
        metricas["AUC-ROC"] = roc_auc_score(y_test, y_prob[:, 1])

    return metricas


def _print_metricas(metricas: dict) -> None:
    for nombre, valor in metricas.items():
        print(f"  {nombre:<12}: {valor:.4f}")
    print()


# ── 1. CARGA Y PREPROCESAMIENTO ───────────────────────────────────────────────

def cargar_y_preprocesar() -> Dataset:
    """
    Carga el CSV, aplica limpieza y preprocesamiento:
      - Imputación de nulos con la mediana (robusta ante outliers)
      - Codificación del target categórico con LabelEncoder
      - División 80/20 train/test con semilla fija
      - Escalado con StandardScaler (ajustado solo sobre train para evitar
        data leakage: el scaler no debe 'ver' estadísticas del test)

    SVM es especialmente sensible a la escala de las features porque su
    función de decisión depende de distancias en el espacio de features.
    """
    print(f"\n{'='*60}")
    print("CARGA Y PREPROCESAMIENTO — SVM")
    print(f"{'='*60}")

    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset: {DATA_PATH.name}  ({df.shape[0]} filas, {df.shape[1]} columnas)")

    # Seleccionar features y target
    X = df[FEATURES].copy()
    y_continuo = df[TARGET].copy()

    # Imputar nulos con la mediana columna a columna
    X = X.fillna(X.median(numeric_only=True))
    y_continuo = y_continuo.fillna(y_continuo.median())

    # Convertir target continuo → categórico (baja / media / alta)
    y_cat = y_continuo.apply(_categorizar_automatizacion).values

    # Codificar etiquetas a enteros (necesario para sklearn)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cat)

    clases = le.classes_
    print(f"  Clases del target: {clases}  ({len(clases)} categorías)")

    feature_names = list(X.columns)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # Escalar: fit solo en train, transform en ambos
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    X_full_sc  = scaler.transform(X)

    print(f"  Train: {X_train_sc.shape[0]} muestras | Test: {X_test_sc.shape[0]} muestras")
    print(f"  Distribución train: { {c: int((y_train == i).sum()) for i, c in enumerate(clases)} }")

    return Dataset(
        X_train_sc=X_train_sc, X_test_sc=X_test_sc,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        classes=clases,
        X_full_sc=X_full_sc, y_full=y_encoded,
        label_encoder=le,
    )


# ── Gráficas compartidas ───────────────────────────────────────────────────────

def _grafica_confusion(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    clases: np.ndarray,
    titulo: str,
    nombre_archivo: str,
) -> None:
    """
    Matriz de confusión normalizada y sin normalizar lado a lado.
    La diagonal principal son los aciertos; el resto son errores.
    La versión normalizada facilita comparar clases con distinto tamaño.
    """
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, subtitulo in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Conteos absolutos", "Proporción por clase real"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=clases, yticklabels=clases,
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(f"{titulo}\n{subtitulo}")

    _save_fig(fig, PLOTS_DIR / nombre_archivo)


def _grafica_curva_aprendizaje(
    modelo: SVC,
    X_full: np.ndarray,
    y_full: np.ndarray,
    titulo: str,
    nombre_archivo: str,
) -> None:
    """
    Curva de aprendizaje con validación cruzada de 5 folds.
    Brecha grande entre train y val → overfitting.
    Ambas métricas bajas → underfitting.
    La banda sombreada representa la desviación estándar entre folds.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X_full, y_full,
        cv=5, scoring="accuracy",
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for scores, color, label in [
        (train_scores, "steelblue",  "Entrenamiento"),
        (val_scores,   "darkorange", "Validación"),
    ]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(train_sizes, mean, "o-", color=color, label=label)
        ax.fill_between(train_sizes, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{titulo} — Curva de aprendizaje")
    ax.legend()
    _save_fig(fig, PLOTS_DIR / nombre_archivo)


def _grafica_decision_2d(
    modelo: SVC,
    X_sc: np.ndarray,
    y: np.ndarray,
    clases: np.ndarray,
    titulo: str,
    nombre_archivo: str,
    feat_idx: tuple = (0, 1),
) -> None:
    """
    Frontera de decisión en 2D usando las dos primeras features escaladas.
    Las regiones coloreadas muestran qué clase predice el modelo en cada zona.
    Los puntos son las muestras reales; el fondo es la frontera aprendida.

    Nota: visualizar en 2D proyecta un espacio de 6 dimensiones, por lo que
    la frontera real en alta dimensión puede ser más compleja.
    """
    i, j = feat_idx
    X2d  = X_sc[:, [i, j]]

    modelo_2d = SVC(**modelo.get_params())
    modelo_2d.fit(X2d, y)

    x_min, x_max = X2d[:, 0].min() - 0.5, X2d[:, 0].max() + 0.5
    y_min, y_max = X2d[:, 1].min() - 0.5, X2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = modelo_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colores_fondo  = ["#AED6F1", "#A9DFBF", "#F9E79F"]
    colores_puntos = ["steelblue", "seagreen", "goldenrod"]
    n_clases = len(clases)

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(n_clases):
        ax.contourf(
            xx, yy, (Z == k).astype(int),
            levels=[0.5, 1.5],
            colors=[colores_fondo[k % len(colores_fondo)]],
            alpha=0.5,
        )

    for k, (clase, color) in enumerate(zip(clases, colores_puntos)):
        mask = y == k
        ax.scatter(
            X2d[mask, 0], X2d[mask, 1],
            c=color, label=clase, edgecolors="white", s=50, alpha=0.8,
        )

    ax.set_xlabel(f"Feature: {modelo.feature_names_in_[i] if hasattr(modelo, 'feature_names_in_') else i}")
    ax.set_ylabel(f"Feature: {modelo.feature_names_in_[j] if hasattr(modelo, 'feature_names_in_') else j}")
    ax.set_title(f"{titulo}\nFrontera de decisión (proyección 2D)")
    ax.legend(title="Clase")
    _save_fig(fig, PLOTS_DIR / nombre_archivo)


# ── 2. SVM KERNEL LINEAL ──────────────────────────────────────────────────────

def ejecutar_svm_lineal(ds: Dataset) -> dict:
    """
    SVM con kernel lineal: separa clases con un hiperplano en el espacio original.
    Es efectivo cuando los datos son (aproximadamente) linealmente separables.

    C controla el trade-off entre margen amplio y errores de clasificación:
    · C grande → margen estrecho, menos errores en train (riesgo de overfitting)
    · C pequeño → margen amplio, permite más errores (mayor generalización)

    Se usa GridSearchCV con 5-fold CV para encontrar el mejor C.
    """
    print("── SVM Kernel Lineal ─────────────────────────────────────")

    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    svc = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    gs  = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(ds.X_train_sc, ds.y_train)

    mejor_modelo = gs.best_estimator_
    print(f"  Mejor C: {gs.best_params_['C']}  (CV accuracy: {gs.best_score_:.4f})")

    y_pred = mejor_modelo.predict(ds.X_test_sc)
    y_prob = mejor_modelo.predict_proba(ds.X_test_sc)

    metricas = _calc_metricas_clasificacion(ds.y_test, y_pred, y_prob)
    _print_metricas(metricas)

    # Gráfica 1: Matriz de confusión
    _grafica_confusion(
        ds.y_test, y_pred, ds.classes,
        "SVM Lineal", "svm_lineal_confusion.png",
    )

    # Gráfica 2: Frontera de decisión 2D
    _grafica_decision_2d(
        mejor_modelo, ds.X_train_sc, ds.y_train, ds.classes,
        "SVM Lineal", "svm_lineal_frontera.png",
    )

    # Gráfica 3: Curva de aprendizaje
    _grafica_curva_aprendizaje(
        mejor_modelo, ds.X_full_sc, ds.y_full,
        "SVM Lineal", "svm_lineal_aprendizaje.png",
    )

    return metricas


# ── 3. SVM KERNEL POLINÓMICO (No lineal 1) ────────────────────────────────────

def ejecutar_svm_polinomico(ds: Dataset) -> dict:
    """
    SVM con kernel polinómico: mapea implícitamente los datos a un espacio
    de mayor dimensión usando polinomios de grado 'degree'.

    K(x, z) = (γ · xᵀz + r)^d

    Captura interacciones no lineales entre features. Útil cuando la frontera
    de decisión tiene forma curva pero estructurada (no caótica).

    degree=3 (cúbico) es el más común: más expresivo que el cuadrático
    pero menos propenso a overfitting que grados más altos.
    """
    print("── SVM Kernel Polinómico (No lineal 1) ───────────────────")

    param_grid = {
        "C":      [0.1, 1, 10],
        "degree": [2, 3, 4],
        "gamma":  ["scale", "auto"],
    }
    svc = SVC(kernel="poly", probability=True, random_state=RANDOM_STATE)
    gs  = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(ds.X_train_sc, ds.y_train)

    mejor_modelo = gs.best_estimator_
    print(f"  Mejores params: {gs.best_params_}  (CV accuracy: {gs.best_score_:.4f})")

    y_pred = mejor_modelo.predict(ds.X_test_sc)
    y_prob = mejor_modelo.predict_proba(ds.X_test_sc)

    metricas = _calc_metricas_clasificacion(ds.y_test, y_pred, y_prob)
    _print_metricas(metricas)

    # Gráfica 1: Matriz de confusión
    _grafica_confusion(
        ds.y_test, y_pred, ds.classes,
        "SVM Polinómico", "svm_polinomico_confusion.png",
    )

    # Gráfica 2: Frontera de decisión 2D
    _grafica_decision_2d(
        mejor_modelo, ds.X_train_sc, ds.y_train, ds.classes,
        "SVM Polinómico", "svm_polinomico_frontera.png",
    )

    # Gráfica 3: Curva de aprendizaje
    _grafica_curva_aprendizaje(
        mejor_modelo, ds.X_full_sc, ds.y_full,
        "SVM Polinómico", "svm_polinomico_aprendizaje.png",
    )

    return metricas


# ── 4. SVM KERNEL SIGMOIDE (No lineal 2) ──────────────────────────────────────

def ejecutar_svm_sigmoide(ds: Dataset) -> dict:
    """
    SVM con kernel sigmoide: análogo a una red neuronal de una capa.

    K(x, z) = tanh(γ · xᵀz + r)

    Este kernel NO garantiza ser definido positivo para todos los valores de
    γ y r, por lo que puede ser menos estable que RBF o Polinómico.
    Es útil cuando se sospecha una relación tipo sigmoide entre features y target.
    """
    print("── SVM Kernel Sigmoide (No lineal 2) ─────────────────────")

    param_grid = {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "coef0": [0.0, 0.5, 1.0],
    }
    svc = SVC(kernel="sigmoid", probability=True, random_state=RANDOM_STATE)
    gs  = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(ds.X_train_sc, ds.y_train)

    mejor_modelo = gs.best_estimator_
    print(f"  Mejores params: {gs.best_params_}  (CV accuracy: {gs.best_score_:.4f})")

    y_pred = mejor_modelo.predict(ds.X_test_sc)
    y_prob = mejor_modelo.predict_proba(ds.X_test_sc)

    metricas = _calc_metricas_clasificacion(ds.y_test, y_pred, y_prob)
    _print_metricas(metricas)

    # Gráfica 1: Matriz de confusión
    _grafica_confusion(
        ds.y_test, y_pred, ds.classes,
        "SVM Sigmoide", "svm_sigmoide_confusion.png",
    )

    # Gráfica 2: Frontera de decisión 2D
    _grafica_decision_2d(
        mejor_modelo, ds.X_train_sc, ds.y_train, ds.classes,
        "SVM Sigmoide", "svm_sigmoide_frontera.png",
    )

    # Gráfica 3: Curva de aprendizaje
    _grafica_curva_aprendizaje(
        mejor_modelo, ds.X_full_sc, ds.y_full,
        "SVM Sigmoide", "svm_sigmoide_aprendizaje.png",
    )

    return metricas


# ── 5. SVM KERNEL RBF / RADIAL ────────────────────────────────────────────────

def ejecutar_svm_rbf(ds: Dataset) -> dict:
    """
    SVM con kernel RBF (Radial Basis Function): el más versátil y usado en
    la práctica. Mide similitud como una función gaussiana de la distancia:

    K(x, z) = exp(-γ · ‖x - z‖²)

    γ controla el radio de influencia de cada punto de soporte:
    · γ grande → radio pequeño, frontera más irregular (riesgo de overfitting)
    · γ pequeño → radio grande, frontera más suave (riesgo de underfitting)

    La combinación C + γ se optimiza con GridSearchCV.
    """
    print("── SVM Kernel RBF (Radial) ───────────────────────────────")

    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 1],
    }
    svc = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    gs  = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(ds.X_train_sc, ds.y_train)

    mejor_modelo = gs.best_estimator_
    print(f"  Mejores params: {gs.best_params_}  (CV accuracy: {gs.best_score_:.4f})")

    y_pred = mejor_modelo.predict(ds.X_test_sc)
    y_prob = mejor_modelo.predict_proba(ds.X_test_sc)

    metricas = _calc_metricas_clasificacion(ds.y_test, y_pred, y_prob)
    _print_metricas(metricas)

    # Gráfica 1: Matriz de confusión
    _grafica_confusion(
        ds.y_test, y_pred, ds.classes,
        "SVM RBF (Radial)", "svm_rbf_confusion.png",
    )

    # Gráfica 2: Frontera de decisión 2D
    _grafica_decision_2d(
        mejor_modelo, ds.X_train_sc, ds.y_train, ds.classes,
        "SVM RBF (Radial)", "svm_rbf_frontera.png",
    )

    # Gráfica 3: Curva de aprendizaje
    _grafica_curva_aprendizaje(
        mejor_modelo, ds.X_full_sc, ds.y_full,
        "SVM RBF (Radial)", "svm_rbf_aprendizaje.png",
    )

    return metricas


# ── 6. COMPARACIÓN FINAL ──────────────────────────────────────────────────────

def graficar_comparacion(resultados: dict) -> None:
    """
    Panel comparativo de los 4 kernels SVM con 5 métricas:
    Accuracy, Precision, Recall, F1 y AUC-ROC.

    Genera dos gráficas:
      · Barras agrupadas: permite comparar métricas entre kernels
      · Radar chart: visión holística del perfil de cada kernel
    """
    kernels  = list(resultados.keys())
    metricas = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    colores  = ["steelblue", "seagreen", "goldenrod", "tomato"]

    # ── Barras agrupadas ──────────────────────────────────────────────────────
    x     = np.arange(len(metricas))
    ancho = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (kernel, color) in enumerate(zip(kernels, colores)):
        vals = [resultados[kernel].get(m, 0) for m in metricas]
        bars = ax.bar(x + i * ancho, vals, ancho, label=kernel, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x + ancho * (len(kernels) - 1) / 2)
    ax.set_xticklabels(metricas)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Comparación de Kernels SVM — Todas las métricas")
    ax.legend(title="Kernel", loc="upper right")
    _save_fig(fig, PLOTS_DIR / "svm_comparacion_barras.png")

    # ── Radar chart ───────────────────────────────────────────────────────────
    n_metricas = len(metricas)
    angulos    = np.linspace(0, 2 * np.pi, n_metricas, endpoint=False).tolist()
    angulos   += angulos[:1]  # Cerrar el polígono

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for kernel, color in zip(kernels, colores):
        vals  = [resultados[kernel].get(m, 0) for m in metricas]
        vals += vals[:1]
        ax.plot(angulos, vals, "o-", color=color, linewidth=2, label=kernel)
        ax.fill(angulos, vals, color=color, alpha=0.1)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Radar — Perfil de métricas por kernel SVM", pad=20)
    ax.legend(title="Kernel", loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save_fig(fig, PLOTS_DIR / "svm_comparacion_radar.png")

    # ── Resumen en consola ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO — KERNELS SVM")
    print(f"{'='*60}")
    print(f"  {'Kernel':<20} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*50}")
    for kernel in kernels:
        m = resultados[kernel]
        print(
            f"  {kernel:<20} "
            f"{m.get('Accuracy', 0):>10.4f} "
            f"{m.get('F1', 0):>10.4f} "
            f"{m.get('AUC-ROC', 0):>10.4f}"
        )

    mejor = max(kernels, key=lambda k: resultados[k].get("Accuracy", 0))
    print(f"\n  Mejor kernel por Accuracy: {mejor}  ({resultados[mejor]['Accuracy']:.4f})")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("ignore")

    print("\n" + "=" * 60)
    print("PIPELINE SVM — Máquinas de Soporte Vectorial")
    print("=" * 60)

    # Crear carpeta de visualizaciones automáticamente
    _preparar_carpeta_visualizaciones()

    # Cargar y preprocesar datos
    ds = cargar_y_preprocesar()

    resultados: dict = {}

    # Ejecutar los 4 kernels
    resultados["Lineal"]     = ejecutar_svm_lineal(ds)
    resultados["Polinómico"] = ejecutar_svm_polinomico(ds)
    resultados["Sigmoide"]   = ejecutar_svm_sigmoide(ds)
    resultados["RBF"]        = ejecutar_svm_rbf(ds)

    # Comparación final
    graficar_comparacion(resultados)

    print(f"Todas las gráficas guardadas en: {PLOTS_DIR}")
    print("Pipeline SVM completado.\n")


if __name__ == "__main__":
    main()