# Autores: Luciana Hoyos Pérez y Santiago Manco Maya

"""
Modelos Supervisados — Luciana Hoyos Pérez y Santiago Manco Maya
Ejecuta los cuatro modelos sobre cualquier dataset CSV.

Uso:
    python supervised.py --data mi_dataset.csv --target columna_objetivo --task classification
    python supervised.py --data mi_dataset.csv --target columna_objetivo --task regression
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
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

# ── Constantes ────────────────────────────────────────────────────────────────
PLOTS_DIR    = Path("plots")
RANDOM_STATE = 42
TEST_SIZE    = 0.2
DPI          = 150
Task = Literal["classification", "regression"]


# ── Estructura de datos para el dataset preprocesado ─────────────────────────

@dataclass
class Dataset:
    """Contiene todas las versiones del dataset necesarias para los modelos."""
    # Sin escalar — para árboles (no son sensibles a la escala)
    X_train: np.ndarray
    X_test:  np.ndarray
    # Escalados con StandardScaler — para modelos lineales
    X_train_sc: np.ndarray
    X_test_sc:  np.ndarray
    y_train: np.ndarray
    y_test:  np.ndarray
    feature_names: list[str]
    # Dataset completo, usado por learning_curve (requiere validación cruzada)
    X_full: np.ndarray
    y_full: np.ndarray


# ── Helpers internos ──────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Guarda y cierra la figura. Centraliza el patrón repetitivo savefig/close/print."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)  # Liberar RAM; sin esto matplotlib acumula figuras abiertas
    print(f"  Gráfica guardada → {path}")


def _plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    path: Path,
    color: str,
) -> None:
    """Gráfico de barras con las top-20 features ordenadas por importancia."""
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
    _save_fig(fig, path)


def _calc_classification_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Calcula accuracy, precision, recall, F1 y AUC-ROC para clasificación.
    Usa promedio "macro" para tratar todas las clases con igual peso,
    lo que es más justo cuando las clases están desbalanceadas.
    """
    multiclass = len(np.unique(y_test)) > 2
    avg = "macro" if multiclass else "binary"

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
    }

    # Para multiclase se usa One-vs-Rest (OvR): AUC por cada clase vs. las demás
    if multiclass:
        metrics["AUC-ROC"] = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )
    else:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_prob[:, 1])

    return metrics


def _calc_regression_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calcula MAE, MSE, RMSE y R² para regresión."""
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "MSE":  mse,
        "RMSE": float(np.sqrt(mse)),
        "R²":   r2_score(y_test, y_pred),
    }


def _print_metrics(metrics: dict) -> None:
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print()


# ── 1. CARGA Y PREPROCESAMIENTO ───────────────────────────────────────────────

def load_and_preprocess(csv_path: str, target_col: str) -> Dataset:
    """
    Carga el CSV y aplica preprocesamiento genérico:
      - Codificación de categóricas con LabelEncoder
      - Imputación de nulos con la mediana (robusta ante outliers)
      - División 80/20 train/test con semilla fija
      - Escalado con StandardScaler (solo ajustado sobre train para evitar data leakage)
    """
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {csv_path}  ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print(f"Variable objetivo: '{target_col}'")
    print(f"{'='*60}\n")

    # Separar X e y ANTES de imputar: si imputáramos sobre el df completo,
    # las estadísticas de y contaminarían X (data leakage).
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.fillna(y.median()).values

    feature_names = list(X.columns)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # fit_transform solo en train; transform en test evita que el modelo
    # "vea" estadísticas del conjunto de prueba durante el entrenamiento.
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    return Dataset(
        X_train=X_train, X_test=X_test,
        X_train_sc=X_train_sc, X_test_sc=X_test_sc,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        X_full=X, y_full=y,
    )


# ── 2. REGRESIÓN LINEAL ───────────────────────────────────────────────────────

def run_linear_regression(ds: Dataset) -> dict:
    """
    Regresión Lineal: predice valores continuos minimizando el error cuadrático.
    Usa datos ESCALADOS porque features en rangos distintos hacen los coeficientes
    incomparables entre sí.
    """
    print("── Regresión Lineal ──────────────────────────────────────")
    model  = LinearRegression()
    model.fit(ds.X_train_sc, ds.y_train)
    y_pred = model.predict(ds.X_test_sc)

    metrics = _calc_regression_metrics(ds.y_test, y_pred)
    _print_metrics(metrics)

    # Si el modelo fuera perfecto, todos los puntos caerían sobre la línea roja (y=x).
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ds.y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    lims = [
        min(ds.y_test.min(), y_pred.min()),
        max(ds.y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=2, label="Ajuste perfecto")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Valores Predichos")
    ax.set_title("Regresión Lineal — Real vs Predicho")
    ax.legend()
    _save_fig(fig, PLOTS_DIR / "linear_regression.png")

    return metrics


# ── 3. REGRESIÓN LOGÍSTICA ────────────────────────────────────────────────────

def run_logistic_regression(ds: Dataset) -> dict:
    """
    Regresión Logística: clasifica usando la función sigmoide sobre combinación lineal.
    Usa datos ESCALADOS; sin escalado el optimizador converge más lento y puede no
    alcanzar la solución óptima dentro del límite de iteraciones.
    """
    print("── Regresión Logística ───────────────────────────────────")
    classes    = np.unique(ds.y_train)
    multiclass = len(classes) > 2

    # max_iter=1000: el valor por defecto (100) frecuentemente no alcanza
    # para datasets con muchas features o clases mal separables.
    model  = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(ds.X_train_sc, ds.y_train)
    y_pred = model.predict(ds.X_test_sc)
    y_prob = model.predict_proba(ds.X_test_sc)

    metrics = _calc_classification_metrics(ds.y_test, y_pred, y_prob)
    _print_metrics(metrics)

    # Gráfica: Matriz de Confusión
    # Diagonal principal = aciertos; todo lo demás = errores
    cm = confusion_matrix(ds.y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Regresión Logística — Matriz de Confusión")
    _save_fig(fig, PLOTS_DIR / "logistic_confusion.png")

    # Gráfica: Curva ROC — solo tiene sentido en clasificación binaria.
    # Muestra el trade-off TPR/FPR al variar el umbral de decisión.
    # La línea punteada es el clasificador aleatorio (AUC = 0.5).
    if not multiclass:
        fpr, tpr, _ = roc_curve(ds.y_test, y_prob[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {metrics['AUC-ROC']:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Regresión Logística — Curva ROC")
        ax.legend()
        _save_fig(fig, PLOTS_DIR / "logistic_roc.png")

    return metrics


# ── 4. ÁRBOL DE DECISIÓN ──────────────────────────────────────────────────────

def run_decision_tree(ds: Dataset, task: Task) -> dict:
    """
    Árbol de Decisión: divide el espacio de features con reglas feature_i <= umbral.
    Usa datos SIN escalar; los splits se basan en umbrales, no en magnitudes.
    max_depth=5 limita el árbol para evitar sobreajuste (overfitting).
    """
    print("── Árbol de Decisión ─────────────────────────────────────")

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
        classes  = None  # plot_tree no necesita nombres de clase en regresión
        metrics  = _calc_regression_metrics(ds.y_test, y_pred)

    _print_metrics(metrics)

    # Gráfica: Estructura del árbol
    # filled=True colorea los nodos según la clase mayoritaria (clasificación)
    # o el valor promedio (regresión).
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=ds.feature_names, class_names=classes,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de Decisión")
    _save_fig(fig, PLOTS_DIR / "decision_tree.png")

    # feature_importances_ mide cuánto reduce cada feature la impureza promedio
    # en todos los nodos donde aparece (criterio Gini o MSE).
    _plot_feature_importance(
        model.feature_importances_, ds.feature_names,
        "Árbol de Decisión — Importancia de características",
        PLOTS_DIR / "dt_feature_importance.png", color="teal",
    )

    return metrics


# ── 5. RANDOM FOREST ──────────────────────────────────────────────────────────

def run_random_forest(ds: Dataset, task: Task) -> dict:
    """
    Random Forest: ensamble de 100 árboles entrenados con bagging.
    Promedia sus predicciones reduciendo varianza sin aumentar sesgo.
    Usa datos SIN escalar (igual que un árbol individual).
    n_jobs=-1 usa todos los núcleos del CPU para acelerar el entrenamiento.
    """
    print("── Random Forest ─────────────────────────────────────────")

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

    # La importancia promediada sobre 100 árboles es más estable y confiable
    # que la de un solo árbol.
    _plot_feature_importance(
        model.feature_importances_, ds.feature_names,
        "Random Forest — Importancia de características",
        PLOTS_DIR / "rf_feature_importance.png", color="forestgreen",
    )

    # Curva de aprendizaje: muestra cómo evoluciona la métrica con más datos.
    # Brecha grande train/val → overfitting. Ambas bajas → underfitting.
    # La banda sombreada es la desviación estándar entre los 5 folds de CV.
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
        ax.fill_between(train_sizes, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Random Forest — Curva de aprendizaje")
    ax.legend()
    _save_fig(fig, PLOTS_DIR / "rf_learning_curve.png")

    return metrics


# ── 6. COMPARACIÓN FINAL ──────────────────────────────────────────────────────

def plot_comparison(results: dict[str, dict], task: Task) -> None:
    """
    Gráfica de barras comparando la métrica principal entre todos los modelos.
    Solo incluye los modelos que calcularon esa métrica concreta.
    """
    metric = "Accuracy" if task == "classification" else "R²"
    models = [name for name, m in results.items() if metric in m]
    values = [results[name][metric] for name in models]

    if not models:
        return

    colors = ["steelblue", "teal", "forestgreen", "darkorange"][: len(models)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_ylim(0, max(values) * 1.15)  # Margen para que las etiquetas no queden recortadas
    ax.set_ylabel(metric)
    ax.set_title(f"Comparación de modelos supervisados — {metric}")
    _save_fig(fig, PLOTS_DIR / "model_comparison.png")

    print(f"\n{'='*60}")
    print(f"  Comparación final ({metric})")
    print(f"{'='*60}")
    for name, val in zip(models, values):
        bar = "█" * int(val * 30)  # Escala el valor a 30 caracteres de ancho máximo
        print(f"  {name:<22} {bar:<32} {val:.4f}")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Suprimir advertencias de convergencia de sklearn que no afectan los resultados
    warnings.filterwarnings("ignore")
    PLOTS_DIR.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Modelos supervisados genéricos")
    parser.add_argument("--data",   required=True, help="Ruta al CSV del dataset")
    parser.add_argument("--target", required=True, help="Nombre de la columna objetivo")
    parser.add_argument("--task",   required=True, choices=["classification", "regression"],
                        help="Tipo de tarea: classification o regression")
    args = parser.parse_args()

    ds = load_and_preprocess(args.data, args.target)

    results: dict[str, dict] = {}

    if args.task == "regression":
        results["Regresión Lineal"]  = run_linear_regression(ds)
    else:
        results["Reg. Logística"]    = run_logistic_regression(ds)

    results["Árbol de Decisión"] = run_decision_tree(ds, args.task)
    results["Random Forest"]     = run_random_forest(ds, args.task)

    plot_comparison(results, args.task)
    print("Todas las gráficas fueron guardadas en la carpeta plots/")


if __name__ == "__main__":
    main()