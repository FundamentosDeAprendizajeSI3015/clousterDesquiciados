# Modelos Supervisados

Este módulo implementa cuatro algoritmos de aprendizaje supervisado aplicados a un dataset en común, permitiendo comparar su rendimiento mediante métricas y visualizaciones.

---

## Contenido

- [Dataset](#dataset)
- [Preprocesamiento](#preprocesamiento)
- [Modelos](#modelos)
  - [Regresión Lineal](#1-regresión-lineal)
  - [Regresión Logística](#2-regresión-logística)
  - [Árbol de Decisión](#3-árbol-de-decisión)
  - [Random Forest](#4-random-forest)
- [Comparación de Métricas](#comparación-de-métricas)
- [Dependencias](#dependencias)

---

## Dataset

El dataset utilizado contiene las siguientes características:

| Columna | Tipo | Descripción |
|--------|------|-------------|
| `feature_1` | Numérica | ... |
| `feature_2` | Numérica | ... |
| `feature_n` | Categórica | ... |
| `target` | Numérica / Binaria | Variable objetivo |

> En el código se completa esta tabla con las columnas reales del dataset una vez definido.

---

## Preprocesamiento

Antes de entrenar los modelos se aplican los siguientes pasos:

1. Eliminación de valores nulos (`dropna` o imputación con media/mediana).
2. Codificación de variables categóricas (`LabelEncoder` / `OneHotEncoder`).
3. Escalado de características numéricas (`StandardScaler`) — requerido para Regresión Lineal y Logística.
4. División en conjunto de entrenamiento y prueba (`train_test_split`, 80/20).

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

---

## Modelos

### 1. Regresión Lineal

**Tipo de problema:** Regresión (predicción de valores continuos).

**¿Cómo funciona?**
Busca la recta (o hiperplano) que minimiza el error cuadrático medio entre los valores reales y los predichos.

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Implementación:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Tabla de Métricas (Ejemplo):**

| Métrica | Valor |
|---------|-------|
| MAE | — |
| MSE | — |
| RMSE | — |
| R² Score | — |

**Gráfica Ejemplo — Valores Reales vs Predichos:**

```
y_real
  │                          ●
  │                     ●  ●
  │               ●  ●
  │          ●  ●
  │     ● ●
  │  ● ●
  └──────────────────────── y_pred
```

> Se espera que los puntos se distribuyan sobre la línea diagonal `y = x` cuanto mejor sea el ajuste.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Regresión Lineal — Real vs Predicho')
plt.tight_layout()
plt.savefig('plots/linear_regression.png')
plt.show()
```

---

### 2. Regresión Logística

**Tipo de problema:** Clasificación binaria (o multiclase).

**¿Cómo funciona?**
Aplica la función sigmoide sobre una combinación lineal de las características para obtener una probabilidad entre 0 y 1.

```
P(y=1|x) = 1 / (1 + e^(−(β₀ + β₁x₁ + ... + βₙxₙ)))
```

**Implementación:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

**Tabla de Métricas (Ejemplo):**

| Métrica | Clase 0 | Clase 1 | Promedio |
|---------|---------|---------|----------|
| Precision | — | — | — |
| Recall | — | — | — |
| F1-Score | — | — | — |
| AUC-ROC | — | — | — |

**Gráfica — Matriz de Confusión:**

```
              Predicho
              0       1
Real   0  [ TN  |  FP ]
       1  [ FN  |  TP ]
```

```python
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Real 0', 'Real 1'])
plt.title('Regresión Logística — Matriz de Confusión')
plt.tight_layout()
plt.savefig('plots/logistic_confusion.png')
plt.show()
```

**Gráfica — Curva ROC:**

```
TPR│          ●───────────────
   │       ●
   │     ●
   │   ●
   │ ●
   ●──────────────────────── FPR
        AUC ≈ 0.XX
```

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Regresión Logística — Curva ROC')
plt.legend()
plt.tight_layout()
plt.savefig('plots/logistic_roc.png')
plt.show()
```

---

### 3. Árbol de Decisión

**Tipo de problema:** Clasificación o Regresión.

**¿Cómo funciona?**
Divide el espacio de características de forma recursiva usando reglas del tipo `feature_i <= umbral`, minimizando impureza (Gini o Entropía para clasificación, MSE para regresión).

```
               [feature_2 <= 3.5]
                /              \
       [feature_1 <= 1.2]    Clase B
         /         \
      Clase A     Clase B
```

**Implementación:**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Tabla de Métricas (Ejemplo):**

| Métrica | Valor |
|---------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |

**Gráfica — Estructura del árbol:**

```python
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=9)
plt.title('Árbol de Decisión')
plt.tight_layout()
plt.savefig('plots/decision_tree.png')
plt.show()
```

**Gráfica — Importancia de características:**

```
feature_2  ████████████████  0.42
feature_1  ████████          0.21
feature_5  ██████            0.16
feature_3  ████              0.11
feature_4  ██                0.10
```

```python
importances = model.feature_importances_
indices = importances.argsort()[::-1]

plt.bar(range(len(importances)), importances[indices], color='teal')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Árbol de Decisión — Importancia de características')
plt.tight_layout()
plt.savefig('plots/dt_feature_importance.png')
plt.show()
```

---

### 4. Random Forest

**Tipo de problema:** Clasificación o Regresión.

**¿Cómo funciona?**
Entrena múltiples árboles de decisión sobre subconjuntos aleatorios del dataset (bagging) y promedia sus predicciones, reduciendo la varianza y mejorando la generalización.

```
Dataset
   │
   ├── Árbol 1 → predicción 1 ─┐
   ├── Árbol 2 → predicción 2 ─┼─→ Votación / Promedio → Predicción final
   ├── Árbol 3 → predicción 3 ─┤
   └── Árbol N → predicción N ─┘
```

**Implementación:**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

**Métricas:**

| Métrica | Valor |
|---------|-------|
| Accuracy | — |
| Precision | — |
| Recall | — |
| F1-Score | — |
| AUC-ROC | — |

**Gráfica — Importancia de características:**

```python
importances = model.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], color='forestgreen')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Random Forest — Importancia de características')
plt.tight_layout()
plt.savefig('plots/rf_feature_importance.png')
plt.show()
```

**Gráfica — Curva de aprendizaje (n_estimators vs accuracy):**

```
Accuracy
  │
1 │              ●──────────────────
  │          ●
  │       ●
  │    ●
  │  ●
  └──────────────────────────────── n_estimators
     10   25   50  100  200  500
```

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Entrenamiento')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validación')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Accuracy')
plt.title('Random Forest — Curva de aprendizaje')
plt.legend()
plt.tight_layout()
plt.savefig('plots/rf_learning_curve.png')
plt.show()
```

---

## Comparación de Métricas

Ejemplo de la tabla resumen con los resultados de los cuatro modelos sobre el mismo conjunto de prueba:

### Clasificación

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Regresión Logística | — | — | — | — | — |
| Árbol de Decisión | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

### Regresión (si aplica)

| Modelo | MAE | MSE | RMSE | R² |
|--------|-----|-----|------|----|
| Regresión Lineal | — | — | — | — |
| Árbol de Decisión (reg) | — | — | — | — |
| Random Forest (reg) | — | — | — | — |

**Gráfica comparativa — Accuracy por modelo:**

```
Accuracy
  │
1 │          ██          ██
  │    ██    ██    ██    ██
  │    ██    ██    ██    ██
  └─────────────────────────
     Lin.  Log.   DT    RF
     Reg.  Reg.
```

```python
models   = ['Log. Reg.', 'Decision Tree', 'Random Forest']
accuracy = [acc_lr, acc_dt, acc_rf]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracy, color=['steelblue', 'teal', 'forestgreen'])
plt.bar_label(bars, fmt='%.3f', padding=3)
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Comparación de modelos supervisados')
plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.show()
```

---

## Dependencias

```txt
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
```

Instalar con:

```bash
pip install -r requirements.txt
```
