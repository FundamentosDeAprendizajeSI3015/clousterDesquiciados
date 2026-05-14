# =============================================================================
# MÓDULO: SVM — Máquinas de Soporte Vectorial
# Autora: Camila Martínez
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# SGDClassifier con loss='modified_huber' replica SVM y soporta predict_proba.
# Nystroem aproxima el mapa de características de kernels no lineales en O(n).
N_COMPONENTS = 300   # dimensiones de la aproximación Nystroem

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
            cv=3,
            scoring="accuracy",
            train_sizes=[0.1, 0.3, 0.6, 1.0],
            n_jobs=-1
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

def ejecutar_svm(df, base_dir, val_df=None, test_df=None):
    """
    Entrena SVM con cuatro kernels.

    Modos:
      - Normal      : df contiene todos los datos; hace split 80/20 internamente.
      - Distribuido : df=train_df, val_df y test_df ya separados;
                      el test externo se usa para evaluación final.
    """
    print("\n" + "="*60)
    print("MODELO SUPERVISADO — SVM")
    print("="*60)

    save_path = os.path.join(base_dir, "visualizaciones", "SVM")
    os.makedirs(save_path, exist_ok=True)

    print("Guardando en:", save_path)

    if val_df is not None and test_df is not None:
        # Modo distribuido: usar splits externos
        le = LabelEncoder()
        le.fit(df[TARGET].astype(str))

        X_train = df[FEATURES].values
        y_train = le.transform(df[TARGET].astype(str))
        X_test  = test_df[FEATURES].values
        y_test  = le.transform(test_df[TARGET].astype(str))
        clases  = le.classes_

        print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")
    else:
        # Modo normal: split 80/20 interno
        X = df[FEATURES].values
        y = LabelEncoder().fit_transform(df[TARGET].values)
        clases = np.unique(df[TARGET].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    resultados = {}

    _sgd = lambda: SGDClassifier(
        loss="modified_huber", max_iter=1000, random_state=42, n_jobs=-1
    )

    # LINEAL — SGD directo, equivalente a SVM lineal
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", _sgd()),
    ])
    resultados["Lineal"] = entrenar_kernel(
        "linear", modelo,
        {"svc__alpha": [1e-4, 1e-3, 1e-2]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # POLINOMICO — Nystroem approx del kernel polinomial + SGD
    modelo = Pipeline([
        ("scaler",   StandardScaler()),
        ("nystroem", Nystroem(kernel="poly", n_components=N_COMPONENTS, random_state=42)),
        ("svc",      _sgd()),
    ])
    resultados["Polinomico"] = entrenar_kernel(
        "poly", modelo,
        {"nystroem__degree": [2, 3], "svc__alpha": [1e-4, 1e-3]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # SIGMOIDE — Nystroem approx del kernel sigmoide + SGD
    modelo = Pipeline([
        ("scaler",   StandardScaler()),
        ("nystroem", Nystroem(kernel="sigmoid", n_components=N_COMPONENTS, random_state=42)),
        ("svc",      _sgd()),
    ])
    resultados["Sigmoide"] = entrenar_kernel(
        "sigmoid", modelo,
        {"nystroem__coef0": [0.0, 1.0], "svc__alpha": [1e-4, 1e-3]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    # RBF — Nystroem approx del kernel RBF + SGD
    modelo = Pipeline([
        ("scaler",   StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", n_components=N_COMPONENTS, random_state=42)),
        ("svc",      _sgd()),
    ])
    resultados["RBF"] = entrenar_kernel(
        "rbf", modelo,
        {"nystroem__gamma": [0.01, 0.1], "svc__alpha": [1e-4, 1e-3]},
        X_train, X_test, y_train, y_test, clases, save_path
    )

    return resultados