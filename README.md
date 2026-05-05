# Proyecto Desquisiados
## Estructura de trabajo
El proyecto está organizado por **ramas de Git**, donde cada rama corresponde a una etapa del pipeline de Machine Learning. Cada rama tiene su propia carpeta con el código de esa etapa.
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

### Carpeta adicional
| Carpeta | Descripción |
|---|---|
| [data/](data/) | Datasets de entrada y salida (no se sube al repositorio) |

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
