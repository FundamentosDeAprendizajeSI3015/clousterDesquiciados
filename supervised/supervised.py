# Autores: Luciana Hoyos Pérez y Santiago Manco Maya

"""
Modelos Supervisados — Luciana Hoyos Pérez y Santiago Manco Maya
Ejecuta los cuatro modelos sobre cualquier dataset CSV.

Uso:
    python supervised.py --data mi_dataset.csv --target columna_objetivo --task classification
    python supervised.py --data mi_dataset.csv --target columna_objetivo --task regression
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. CARGA Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"Dataset: {csv_path}  ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print(f"Variable objetivo: '{target_col}'")
    print(f"{'='*60}\n")

    # Separar X e y antes de imputar para no contaminar
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # Codificar columnas categóricas en X
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Imputar nulos con la mediana de cada columna
    X = X.fillna(X.median(numeric_only=True))

    # Codificar y si es categórica (clasificación con etiquetas de texto)
    if y.dtype == object or str(y.dtype) == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.fillna(y.median()).values

    feature_names = list(X.columns)
    X = X.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, X, y


# ---------------------------------------------------------------------------
# 2. REGRESIÓN LINEAL
# ---------------------------------------------------------------------------

def run_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    print("── Regresión Lineal ──────────────────────────────────────")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE      : {mae:.4f}")
    print(f"  MSE      : {mse:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  R² Score : {r2:.4f}\n")

    # Gráfica: Valores Reales vs Predichos
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Ajuste perfecto")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Valores Predichos")
    ax.set_title("Regresión Lineal — Real vs Predicho")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/linear_regression.png", dpi=150)
    plt.close(fig)
    print("  Gráfica guardada → plots/linear_regression.png")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


# ---------------------------------------------------------------------------
# 3. REGRESIÓN LOGÍSTICA
# ---------------------------------------------------------------------------

def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("── Regresión Logística ───────────────────────────────────")
    classes = np.unique(y_train)
    multiclass = len(classes) > 2

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    avg = "macro" if multiclass else "binary"
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

    if multiclass:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    else:
        auc = roc_auc_score(y_test, y_prob[:, 1])

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}\n")

    # Gráfica: Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Regresión Logística — Matriz de Confusión")
    fig.tight_layout()
    fig.savefig("plots/logistic_confusion.png", dpi=150)
    plt.close(fig)
    print("  Gráfica guardada → plots/logistic_confusion.png")

    # Gráfica: Curva ROC (solo binaria)
    if not multiclass:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Regresión Logística — Curva ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig("plots/logistic_roc.png", dpi=150)
        plt.close(fig)
        print("  Gráfica guardada → plots/logistic_roc.png")

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC-ROC": auc}


# ---------------------------------------------------------------------------
# 4. ÁRBOL DE DECISIÓN
# ---------------------------------------------------------------------------

def run_decision_tree(X_train, X_test, y_train, y_test, feature_names, task):
    print("── Árbol de Decisión ─────────────────────────────────────")

    if task == "classification":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        model = DecisionTreeRegressor(max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "classification":
        classes = [str(c) for c in np.unique(y_train)]
        multiclass = len(classes) > 2
        avg = "macro" if multiclass else "binary"
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}\n")
        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    else:
        classes = None
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        print(f"  MAE      : {mae:.4f}")
        print(f"  MSE      : {mse:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"  R² Score : {r2:.4f}\n")
        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

    # Gráfica: Estructura del árbol
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=feature_names, class_names=classes,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Árbol de Decisión")
    fig.tight_layout()
    fig.savefig("plots/decision_tree.png", dpi=150)
    plt.close(fig)
    print("  Gráfica guardada → plots/decision_tree.png")

    # Gráfica: Importancia de características
    _plot_feature_importance(
        model.feature_importances_, feature_names,
        "Árbol de Decisión — Importancia de características",
        "plots/dt_feature_importance.png", color="teal"
    )
    print("  Gráfica guardada → plots/dt_feature_importance.png")

    return metrics


# ---------------------------------------------------------------------------
# 5. RANDOM FOREST
# ---------------------------------------------------------------------------

def run_random_forest(X_train, X_test, y_train, y_test, feature_names, task, X_full, y_full):
    print("── Random Forest ─────────────────────────────────────────")

    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "classification":
        multiclass = len(np.unique(y_train)) > 2
        avg = "macro" if multiclass else "binary"
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
        y_prob = model.predict_proba(X_test)
        if multiclass:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        else:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}\n")
        metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC-ROC": auc}
    else:
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        print(f"  MAE      : {mae:.4f}")
        print(f"  MSE      : {mse:.4f}")
        print(f"  RMSE     : {rmse:.4f}")
        print(f"  R² Score : {r2:.4f}\n")
        metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

    # Gráfica: Importancia de características
    _plot_feature_importance(
        model.feature_importances_, feature_names,
        "Random Forest — Importancia de características",
        "plots/rf_feature_importance.png", color="forestgreen"
    )
    print("  Gráfica guardada → plots/rf_feature_importance.png")

    # Gráfica: Curva de aprendizaje
    scoring = "accuracy" if task == "classification" else "r2"
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_full, y_full, cv=5, scoring=scoring,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="forestgreen", label="Entrenamiento")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="forestgreen")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="darkorange", label="Validación")
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="darkorange")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Random Forest — Curva de aprendizaje")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/rf_learning_curve.png", dpi=150)
    plt.close(fig)
    print("  Gráfica guardada → plots/rf_learning_curve.png")

    return metrics


# ---------------------------------------------------------------------------
# 6. COMPARACIÓN FINAL
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, task: str):
    if task == "classification":
        metric = "Accuracy"
    else:
        metric = "R²"

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
    fig.tight_layout()
    fig.savefig("plots/model_comparison.png", dpi=150)
    plt.close(fig)
    print("\n  Gráfica guardada → plots/model_comparison.png")

    print(f"\n{'='*60}")
    print(f"  Comparación final ({metric})")
    print(f"{'='*60}")
    for name, val in zip(models, values):
        bar = "█" * int(val * 30)
        print(f"  {name:<22} {bar:<32} {val:.4f}")
    print()


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _plot_feature_importance(importances, feature_names, title, path, color):
    indices = importances.argsort()[::-1]
    top_n = min(20, len(importances))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices[:top_n]], color=color)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices[:top_n]], rotation=45, ha="right")
    ax.set_ylabel("Importancia")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Modelos supervisados genéricos")
    parser.add_argument("--data",   required=True, help="Ruta al CSV del dataset")
    parser.add_argument("--target", required=True, help="Nombre de la columna objetivo")
    parser.add_argument("--task",   required=True, choices=["classification", "regression"],
                        help="Tipo de tarea: classification o regression")
    args = parser.parse_args()

    (X_train, X_test,
     X_train_sc, X_test_sc,
     y_train, y_test,
     feature_names,
     X_full, y_full) = load_and_preprocess(args.data, args.target)

    results = {}

    if args.task == "regression":
        results["Regresión Lineal"] = run_linear_regression(
            X_train_sc, X_test_sc, y_train, y_test, feature_names
        )
    else:
        results["Reg. Logística"] = run_logistic_regression(
            X_train_sc, X_test_sc, y_train, y_test
        )

    results["Árbol de Decisión"] = run_decision_tree(
        X_train, X_test, y_train, y_test, feature_names, args.task
    )

    results["Random Forest"] = run_random_forest(
        X_train, X_test, y_train, y_test, feature_names,
        args.task, X_full, y_full
    )

    plot_comparison(results, args.task)
    print("Todas las gráficas fueron guardadas en la carpeta plots/")


if __name__ == "__main__":
    main()
