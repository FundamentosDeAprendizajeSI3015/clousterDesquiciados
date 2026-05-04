# =============================================================================
# MÓDULO: SVM — Máquinas de Soporte Vectorial
# Autora: Camila Martínez
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

FEATURES = ["exp", "rutina", "estructuracion", "creatividad", "resolucion", "interaccion"]
TARGET = "automatizacion_cat"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _guardar_figura(fig, save_path, nombre):
    os.makedirs(save_path, exist_ok=True)
    ruta = os.path.join(save_path, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Guardado: {ruta}")


def calcular_metricas(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC-ROC": auc
    }


# -----------------------------------------------------------------------------
# GRÁFICAS
# -----------------------------------------------------------------------------

def graficar_confusion(y_true, y_pred, clases, save_path, nombre):
    fig, ax = plt.subplots(figsize=(6,5))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=ax)

    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

    _guardar_figura(fig, save_path, nombre)


def graficar_curva(modelo, X, y, save_path, nombre):
    try:
        sizes, train_scores, val_scores = learning_curve(
            modelo, X, y,
            cv=5,
            scoring="accuracy",
            train_sizes=[0.1, 0.3, 0.6, 1.0],
            n_jobs=1
        )

        fig, ax = plt.subplots(figsize=(7,5))

        ax.plot(sizes, train_scores.mean(axis=1), label="Train")
        ax.plot(sizes, val_scores.mean(axis=1), label="Validación")

        ax.set_title("Curva de aprendizaje")
        ax.set_xlabel("Tamaño entrenamiento")
        ax.set_ylabel("Accuracy")
        ax.legend()

        _guardar_figura(fig, save_path, nombre)

    except Exception as e:
        print(f"Error curva: {e}")


# -----------------------------------------------------------------------------
# ENTRENAMIENTO
# -----------------------------------------------------------------------------

def entrenar_kernel(nombre, modelo, params, X_train, X_test, y_train, y_test, clases, save_path):

    print(f"\n--- SVM {nombre} ---")

    gs = GridSearchCV(modelo, params, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    metricas = calcular_metricas(y_test, y_pred, y_prob)

    for k, v in metricas.items():
        print(f"{k}: {v:.4f}")

    graficar_confusion(y_test, y_pred, clases, save_path, f"{nombre}_confusion.png")
    graficar_curva(best_model, X_train, y_train, save_path, f"{nombre}_learning.png")

    return metricas


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------

def ejecutar_svm(df, base_dir):

    print("\n" + "="*60)
    print("MODELO SUPERVISADO — SVM")
    print("="*60)

    save_path = os.path.join(base_dir, "visualizaciones", "SVM")
    os.makedirs(save_path, exist_ok=True)

    print("Guardando en:", save_path)

    X = df[FEATURES].values
    y = LabelEncoder().fit_transform(df[TARGET].values)

    clases = np.unique(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    resultados = {}

    # LINEAL
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", probability=True))
    ])
    resultados["Lineal"] = entrenar_kernel(
        "linear", modelo, {"svc__C":[0.1,1,10]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # POLINOMICO
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="poly", probability=True))
    ])
    resultados["Polinomico"] = entrenar_kernel(
        "poly", modelo,
        {"svc__C":[0.1,1], "svc__degree":[2,3]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # SIGMOIDE
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="sigmoid", probability=True))
    ])
    resultados["Sigmoide"] = entrenar_kernel(
        "sigmoid", modelo,
        {"svc__C":[0.1,1], "svc__coef0":[0,1]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # RBF
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])
    resultados["RBF"] = entrenar_kernel(
        "rbf", modelo,
        {"svc__C":[0.1,1,10], "svc__gamma":["scale",0.1]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    return resultados