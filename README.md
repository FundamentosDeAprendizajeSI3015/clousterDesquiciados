# Proyecto Desquisiados
## Estructura de trabajo
---
### El proyecto está organizado por **ramas de Git**, donde cada rama corresponde a una etapa del pipeline de Machine Learning. Cada rama tiene su propia carpeta con el código de esa etapa.
---
## Ramas y carpetas
Cada rama tiene su propio archivo de documentación (p. ej. `readme_eda.md`, `readme_supervisado.md`) dentro de su carpeta.
| Rama | Carpeta | Descripción | Integrantes |
|---|---|---|---|
| `load_data` | [load_data/](load_data/) | Carga y limpieza del dataset | Mariana Valderrama, Alexandra Hurtado |
| `eda` | [eda/](eda/) | Análisis Exploratorio de Datos + gráficas | Mariana Valderrama, Alexandra Hurtado |
| `no_supervisado` | [no_supervisado/](no_supervisado/) | Modelos de clustering + scores | Alejandra Ocampo |
| `supervisado` | [supervisado/](supervisado/) | Modelos supervisados + scores | Santiago Manco, Luciana Hoyos |
| `svm` | [svm/](svm/) | Máquinas de Soporte Vectorial | Camila Martínez |
| `visualizacion` | [visualizacion/](visualizacion/) | Gráficas de todos los módulos (EDA, supervisado y no supervisado) | Todos |
| `desarrollo` | [Desarrollo/](Desarrollo/) | Pipeline principal que integra todas las etapas | Todos |
| `produccion` | [Produccion/](Produccion/) | Versión de despliegue en producción | Todos |

> Las gráficas generadas por cada módulo deben guardarse en [visualizacion/graficas/](visualizacion/graficas/).
> Los scores y métricas deben guardarse en [index_score/](index_score/).

### Carpetas adicionales
| Carpeta | Descripción |
|---|---|
| [data/](data/) | Datasets de entrada y salida (no se sube al repositorio) |
| [mkl/](mkl/) | Módulo de Multiple Kernel Learning con Extremality Ordering |

---
## Módulo MKL — Multiple Kernel Learning

El módulo [`mkl/`](mkl/) implementa el algoritmo **Extremality Multiple Kernel Learning (EMKL)**, basado en el repositorio [extremality_mkl](https://github.com/maospina1041/extremality_mkl), y adaptado al dataset del proyecto.

### ¿Qué hace?

1. Genera múltiples **kernels polinomiales débiles** con subconjuntos aleatorios de features.
2. Calcula métricas de calidad para cada kernel: **Alignment**, **Polarization**, **FSM** y **Complex Ratio**.
3. Asigna pesos a los kernels usando **Extremality Ordering** (ordenamiento geométrico en espacio de métricas).
4. Compara cuatro estrategias de combinación: **Natural**, **Anti-Natural**, **RBF** y **Polinomial**.
5. Genera gráficas de evolución de métricas guardadas en `visualizaciones/mkl/`.

### Estructura del módulo

| Archivo | Descripción |
|---|---|
| `extremality_order.py` | Gram-Schmidt + rotación de matrices + orden de extremalidad |
| `kernel_metrics.py` | Métricas de calidad de kernels (Alignment, Polarization, FSM, Complex Ratio) |
| `weak_polynomial_kernel.py` | Generación de kernels polinomiales débiles con selección aleatoria de features |
| `weight_linear_combination.py` | Normalización y potenciación de pesos |
| `extremality_weights.py` | Cálculo de pesos `w_1` (Natural) y `w_2` (Anti-Natural) |
| `mkl_simulation.py` | Simulación con splits aleatorios y evaluación de los 4 métodos |
| `mkl_plots.py` | Gráficas de métricas vs. número de kernels + heatmap de resultados |
| `mkl_pipeline.py` | Función `ejecutar_mkl()` que orquesta todo e integra con `pipeline.py` |

### Paso en el pipeline

Es el **paso 14**, se ejecuta tras los modelos supervisados:

```python
ejecutar_mkl(
    df,
    features=FEATURES,
    target_col="automatizacion_cat",
    save_path="visualizaciones/mkl/",
    n_iter=10,
    kernels_list=[5, 10, 20],
    t=4,
)
```

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `n_iter` | `10` | Iteraciones de train/test split por configuración |
| `kernels_list` | `[5, 10, 20]` | Número de kernels débiles a evaluar |
| `t` | `4` | Máximo de features por kernel (≤ 6 features del dataset) |

Las gráficas se guardan automáticamente en `visualizaciones/mkl/`.

---
## Integrantes y responsabilidades
| Nombre | Responsabilidad |
|---|---|
| Mariana Valderrama | Carga de datos, EDA y gráficas |
| Alexandra Hurtado | Carga de datos, EDA y gráficas |
| Santiago Manco | Modelos supervisados, gráficas y scores |
| Luciana Hoyos | Modelos supervisados, gráficas y scores |
| Camila Martínez | Máquinas de Soporte Vectorial |
| Alejandra Ocampo | Modelos no supervisados, gráficas y scores |

---
## Cómo ejecutar el pipeline completo
```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn scipy
cd Desarrollo
python pipeline.py
```
