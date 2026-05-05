# =============================================================================
# MÓDULO: Modelos Supervisados
# Autores: Luciana Hoyos Pérez y Santiago Manco Maya
# =============================================================================

import os
import warnings
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TEST_SIZE    = 0.2
DPI          = 150
Task = Literal["classification", "regression"]


# -----------------------------------------------------------------------------
# Estructura interna del dataset
# -----------------------------------------------------------------------------

@dataclass
class Dataset:
    X_train:      np.ndarray
    X_test:       np.ndarray
    X_train_sc:   np.ndarray
    X_test_sc:    np.ndarray
    y_train:      np.ndarray
    y_test:       np.ndarray
    feature_names: list
    X_full:       np.ndarray
    y_full:       np.ndarray


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _guardar(save_path, nombre_archivo):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ruta = os.path.join(save_path, nombre_archivo)
        plt.savefig(ruta, bbox_inches='tight', dpi=DPI)
        print(f"  Guardado: {ruta}")


def _save_fig(fig, save_path, nombre_archivo):
    fig.tight_layout()
    _guardar(save_path, nombre_archivo)
    plt.close(fig)


def _plot_feature_importance(importances, feature_names, title, save_path,
                              nombre_archivo, color):
    indices = importances.argsort()[::-1]
    top_n   = min(20, len(importances))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices[:top_n]], color=color)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(
        [feature_names[i] for i in indices[:top_n]], rotation=45, ha="right"
    )
    ax.set_ylabel("Importancia")
    ax.set_title(title)
    _save_fig(fig, save_path, nombre_archivo)


def _calc_classification_metrics(y_test, y_pred, y_prob):
    multiclass = len(np.unique(y_test)) > 2
    avg = "macro" if multiclass else "binary"
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
    }
    if multiclass:
        metrics["AUC-ROC"] = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )
    else:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_prob[:, 1])
    return metrics


def _calc_regression_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "MSE":  mse,
        "RMSE": float(np.sqrt(mse)),
        "R²":   r2_score(y_test, y_pred),
    }


def _print_metrics(metrics):
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print()


# -----------------------------------------------------------------------------
# 1. Carga y preprocesamiento
# -----------------------------------------------------------------------------

def cargar_y_preprocesar(df_or_path, target_col, features=None):
    """
    Acepta un DataFrame o ruta a CSV. Aplica codificación, imputación,
    split 80/20 y escalado con StandardScaler.
    Si se pasa `features`, se usan solo esas columnas como X.
    """
    print("\n" + "="*60)
    print("MODELOS SUPERVISADOS")
    print("="*60)

    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        df = pd.read_csv(df_or_path)

    print(f"\n  Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  Variable objetivo: '{target_col}'")

    if features is not None:
        X = df[features].copy()
    else:
        X = df.drop(columns=[target_col]).copy()

    y = df[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    if pd.api.types.is_numeric_dtype(y):
        y = y.fillna(y.median()).values
    else:
        y = LabelEncoder().fit_transform(y.astype(str))

    feature_names = list(X.columns)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return Dataset(
        X_train=X_train, X_test=X_test,
        X_train_sc=X_train_sc, X_test_sc=X_test_sc,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        X_full=X, y_full=y,
    )


# -----------------------------------------------------------------------------
# 2. Regresión Lineal
# -----------------------------------------------------------------------------

def ejecutar_regresion_lineal(ds, save_path=None):
    """Regresión Lineal sobre datos escalados. Gráfica: Real vs Predicho."""
    print("\n[Sup.1] Regresión Lineal...")
    model  = LinearRegression()
    model.fit(ds.X_train_sc, ds.y_train)
    y_pred = model.predict(ds.X_test_sc)

    metrics = _calc_regression_metrics(ds.y_test, y_pred)
    _print_metrics(metrics)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ds.y_test, y_pred, alpha=0.6, color="steelblue",
               edgecolors="white", s=60)
    lims = [
        min(ds.y_test.min(), y_pred.min()),
        max(ds.y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=2, label="Ajuste perfecto")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Valores Predichos")
    ax.set_title("Regresión Lineal — Real vs Predicho")
    ax.legend()
    _save_fig(fig, save_path, "linear_regression.png")

    return metrics


# -----------------------------------------------------------------------------
# 3. Regresión Logística
# -----------------------------------------------------------------------------

def ejecutar_regresion_logistica(ds, save_path=None):
    """Regresión Logística sobre datos escalados. Gráficas: CM y ROC."""
    print("\n[Sup.2] Regresión Logística...")
    classes    = np.unique(ds.y_train)
    multiclass = len(classes) > 2

    model  = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(ds.X_train_sc, ds.y_train)
    y_pred = model.predict(ds.X_test_sc)
    y_prob = model.predict_proba(ds.X_test_sc)

    metrics = _calc_classification_metrics(ds.y_test, y_pred, y_prob)
    _print_metrics(metrics)

    # Matriz de confusión
    cm_vals = confusion_matrix(ds.y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_vals, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Regresión Logística — Matriz de Confusión")
    _save_fig(fig, save_path, "logistic_confusion.png")

    # Curva ROC (solo binaria)
    if not multiclass:
        fpr, tpr, _ = roc_curve(ds.y_test, y_prob[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"AUC = {metrics['AUC-ROC']:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Regresión Logística — Curva ROC")
        ax.legend()
        _save_fig(fig, save_path, "logistic_roc.png")

    return metrics


# -----------------------------------------------------------------------------
# 4. Árbol de Decisión
# -----------------------------------------------------------------------------

def ejecutar_arbol_decision(ds, task, save_path=None):
    """Árbol de Decisión (max_depth=5) sobre datos sin escalar."""
    print("\n[Sup.3] Árbol de Decisión...")

    if task == "classification":
        model = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    else:
        model = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)

    model.fit(ds.X_train, ds.y_train)
    y_pred = model.predict(ds.X_test)

    if task == "classification":
        classes  = [str(c) for c in np.unique(ds.y_train)]
        y_prob   = model.predict_proba(ds.X_test)
        metrics  = _calc_classification_metrics(ds.y_test, y_pred, y_prob)
    else:
        classes  = None
        metrics  = _calc_regression_metrics(ds.y_test, y_pred)

    _print_metrics(metrics)

    # Estructura del árbol
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=ds.feature_names, class_names=classes,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de Decisión")
    _save_fig(fig, save_path, "decision_tree.png")

    _plot_feature_importance(
        model.feature_importances_, ds.feature_names,
        "Árbol de Decisión — Importancia de características",
        save_path, "dt_feature_importance.png", color="teal",
    )

    return metrics


# -----------------------------------------------------------------------------
# 5. Random Forest
# -----------------------------------------------------------------------------

def ejecutar_random_forest(ds, task, save_path=None):
    """Random Forest (100 árboles) con curva de aprendizaje."""
    print("\n[Sup.4] Random Forest...")

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        )

    model.fit(ds.X_train, ds.y_train)
    y_pred = model.predict(ds.X_test)

    if task == "classification":
        y_prob  = model.predict_proba(ds.X_test)
        metrics = _calc_classification_metrics(ds.y_test, y_pred, y_prob)
    else:
        metrics = _calc_regression_metrics(ds.y_test, y_pred)

    _print_metrics(metrics)

    _plot_feature_importance(
        model.feature_importances_, ds.feature_names,
        "Random Forest — Importancia de características",
        save_path, "rf_feature_importance.png", color="forestgreen",
    )

    # Curva de aprendizaje
    scoring = "accuracy" if task == "classification" else "r2"
    train_sizes, train_scores, val_scores = learning_curve(
        model, ds.X_full, ds.y_full, cv=5, scoring=scoring,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for scores, color, label in [
        (train_scores, "forestgreen", "Entrenamiento"),
        (val_scores,   "darkorange",  "Validación"),
    ]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(train_sizes, mean, "o-", color=color, label=label)
        ax.fill_between(train_sizes, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Random Forest — Curva de aprendizaje")
    ax.legend()
    _save_fig(fig, save_path, "rf_learning_curve.png")

    return metrics


# -----------------------------------------------------------------------------
# 6. Comparación final
# -----------------------------------------------------------------------------

def graficar_comparacion(results, task, save_path=None):
    """Barras comparando la métrica principal entre todos los modelos."""
    metric = "Accuracy" if task == "classification" else "R²"
    models = [name for name, m in results.items() if metric in m]
    values = [results[name][metric] for name in models]

    if not models:
        return

    colors = ["steelblue", "teal", "forestgreen", "darkorange"][: len(models)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparación de modelos supervisados — {metric}")
    _save_fig(fig, save_path, "model_comparison.png")

    print(f"\n{'='*60}")
    print(f"  Comparación final ({metric})")
    print(f"{'='*60}")
    for name, val in zip(models, values):
        bar = "█" * int(val * 30)
        print(f"  {name:<22} {bar:<32} {val:.4f}")
    print()


# -----------------------------------------------------------------------------
# Pipeline Supervisado
# -----------------------------------------------------------------------------

def ejecutar_supervisado(df_or_path, target_col, task,
                          features=None, save_path=None):
    """
    Pipeline completo de modelos supervisados:
    1. Carga y preprocesamiento
    2. Regresión Lineal (si task='regression') o Logística (si 'classification')
    3. Árbol de Decisión
    4. Random Forest
    5. Comparación final

    Retorna dict con métricas de cada modelo.
    """
    ds = cargar_y_preprocesar(df_or_path, target_col, features)

    results = {}

    if task == "regression":
        results["Regresión Lineal"]  = ejecutar_regresion_lineal(ds, save_path)
    else:
        results["Reg. Logística"]    = ejecutar_regresion_logistica(ds, save_path)

    results["Árbol de Decisión"] = ejecutar_arbol_decision(ds, task, save_path)
    results["Random Forest"]     = ejecutar_random_forest(ds, task, save_path)

    graficar_comparacion(results, task, save_path)

    print(f"\n  Gráficas guardadas en: {save_path}")
    return results


# -----------------------------------------------------------------------------
# Ejecución directa (uso standalone)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modelos supervisados genéricos")
    parser.add_argument("--data",   required=True, help="Ruta al CSV del dataset")
    parser.add_argument("--target", required=True, help="Columna objetivo")
    parser.add_argument("--task",   required=True,
                        choices=["classification", "regression"])
    parser.add_argument("--save_path", default="plots",
                        help="Carpeta donde guardar las gráficas")
    args = parser.parse_args()

    ejecutar_supervisado(
        args.data, args.target, args.task,
        save_path=args.save_path,
    )
