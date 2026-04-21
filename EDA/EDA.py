# =============================================================================
# MÓDULO: Análisis Exploratorio de Datos (EDA)
# Autores: Mariana Valderrama, Alexandra Hurtado
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 3.1  INSPECCIÓN INICIAL
# =============================================================================

def inspeccion_inicial(df):
    """
    Muestra un resumen estructural del DataFrame:
    primeras filas, dimensiones, tipos, nulos y duplicados.

    Parámetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame  Tabla de estadísticas generales (describe).
    """
    print("\n" + "=" * 60)
    print("3.1  INSPECCIÓN INICIAL")
    print("=" * 60)

    # TODO: imprimir primeras filas, dimensiones, tipos, nulos y duplicados

    return df.describe(include="all")


# =============================================================================
# 3.2  MEDIDAS DE TENDENCIA CENTRAL
# Nos indican el valor típico o representativo de cada variable.
# =============================================================================

def tendencia_central(df):
    """
    Imprime media, mediana y moda de las variables numéricas del DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
    """
    print("\n" + "=" * 60)
    print("3.2  MEDIDAS DE TENDENCIA CENTRAL")
    print("=" * 60)
    print("  Nos indican el valor típico de las variables.\n")

    # TODO: iterar sobre columnas numéricas e imprimir media, mediana y moda


# =============================================================================
# 3.3  MEDIDAS DE DISPERSIÓN
# Nos indican qué tan dispersos o concentrados están los datos.
# =============================================================================

def dispersion(df):
    """
    Imprime desviación estándar, varianza, rango e IQR
    de las variables numéricas del DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
    """
    print("\n" + "=" * 60)
    print("3.3  MEDIDAS DE DISPERSIÓN")
    print("=" * 60)
    print("  Nos indican qué tan dispersos están los datos.\n")

    # TODO: iterar sobre columnas numéricas e imprimir desv. std, varianza, rango e IQR


# =============================================================================
# 3.4  TABLA RESUMEN COMPLETA
# Consolida tendencia central y dispersión en un solo DataFrame.
# =============================================================================

def estadisticas_descriptivas(df):
    """
    Calcula en una sola tabla: media, mediana, moda, desv. estándar,
    varianza, coeficiente de variación, rango, asimetría, curtosis,
    mín, Q1, Q3, máx e IQR.

    Parámetros
    ----------
    df : pd.DataFrame

    Retorna
    -------
    pd.DataFrame  Una fila por variable numérica.
    """
    print("\n" + "=" * 60)
    print("3.4  ESTADÍSTICAS DESCRIPTIVAS — TABLA RESUMEN")
    print("=" * 60)

    # TODO: construir DataFrame de stats con todas las métricas y retornarlo


# =============================================================================
# 3.5  VISUALIZACIONES
# =============================================================================

# -----------------------------------------------------------------------------
# Boxplots
# Permiten visualizar la distribución y detectar outliers por variable.
# -----------------------------------------------------------------------------

def graficar_boxplots(df, save_path=None):
    """
    Genera un boxplot por cada variable numérica del DataFrame.

    Parámetros
    ----------
    df        : pd.DataFrame
    save_path : str | None  Carpeta donde guardar la imagen.
    """
    # TODO: crear subplots y graficar un boxplot por columna numérica

    _guardar(save_path, "boxplots.png")
    plt.show()


# -----------------------------------------------------------------------------
# Histogramas
# Permiten observar la distribución de frecuencia de cada variable.
# -----------------------------------------------------------------------------

def graficar_histogramas(df, bins=5, save_path=None):
    """
    Genera un histograma por cada variable numérica del DataFrame.

    Parámetros
    ----------
    bins : int  Número de intervalos (ajustar según los datos).
    """
    # TODO: crear subplots y graficar un histograma por columna numérica

    _guardar(save_path, "histogramas.png")
    plt.show()


# -----------------------------------------------------------------------------
# Matriz de correlación
# Permite identificar relaciones lineales entre variables.
# -----------------------------------------------------------------------------

def graficar_correlacion(df, save_path=None):
    """
    Mapa de calor de la matriz de correlación (triángulo inferior).
    """
    # TODO: calcular corr(), aplicar máscara triangular y graficar heatmap

    _guardar(save_path, "correlacion.png")
    plt.show()


# -----------------------------------------------------------------------------
# Gráfico de dispersión
# Observamos la relación entre dos variables específicas.
# -----------------------------------------------------------------------------

def graficar_dispersion(df, x, y, save_path=None):
    """
    Scatter plot entre dos variables numéricas.

    Parámetros
    ----------
    x : str  Variable en el eje X.
    y : str  Variable en el eje Y.
    """
    # TODO: validar que x e y existen en df y graficar scatter

    _guardar(save_path, f"dispersion_{x}_vs_{y}.png")
    plt.show()


# -----------------------------------------------------------------------------
# Distribución de la variable objetivo (target)
# -----------------------------------------------------------------------------

def graficar_target(df, target_col, save_path=None):
    """
    Muestra la distribución de la variable objetivo.
    - Si tiene pocas categorías: barras + pastel.
    - Si es continua: histograma + KDE.

    Parámetros
    ----------
    target_col : str  Nombre de la columna objetivo.
    """
    # TODO: detectar si el target es categórico o continuo y graficar en consecuencia

    _guardar(save_path, "target_distribucion.png")
    plt.show()


# -----------------------------------------------------------------------------
# Pairplot
# Relaciones cruzadas entre las primeras variables numéricas.
# -----------------------------------------------------------------------------

def graficar_pairplot(df, target_col, max_vars=6, save_path=None):
    """
    Pairplot de hasta `max_vars` variables numéricas, coloreado por target.

    Parámetros
    ----------
    max_vars : int  Máximo de variables a incluir (evita gráficos muy densos).
    """
    # TODO: seleccionar columnas, convertir target a string y llamar sns.pairplot

    _guardar(save_path, "pairplot.png")
    plt.show()


# =============================================================================
# 3.6  PIPELINE EDA COMPLETO
# Punto de entrada único que llama a todas las funciones en orden.
# =============================================================================

def ejecutar_eda(df, target_col, save_path=None):
    """
    Ejecuta el EDA completo en el orden correcto:
        1. Inspección inicial
        2. Tendencia central
        3. Dispersión
        4. Tabla resumen
        5. Boxplots
        6. Histogramas
        7. Matriz de correlación
        8. Dispersión entre variables clave
        9. Distribución del target
        10. Pairplot

    Parámetros
    ----------
    df         : pd.DataFrame
    target_col : str   Nombre de la variable objetivo.
    save_path  : str   Carpeta donde guardar los gráficos generados.

    Retorna
    -------
    dict  Con claves 'describe' y 'stats' para uso en desarrollo.py.
    """
    desc  = inspeccion_inicial(df)
    tendencia_central(df)
    dispersion(df)
    stats = estadisticas_descriptivas(df)

    print("\n[3.5] Generando visualizaciones...")
    graficar_boxplots(df, save_path)
    graficar_histogramas(df, save_path=save_path)
    graficar_correlacion(df, save_path)
    graficar_dispersion(df, save_path=save_path)      # TODO: definir x e y
    graficar_target(df, target_col, save_path)
    graficar_pairplot(df, target_col, save_path=save_path)

    return {"describe": desc, "stats": stats}


# =============================================================================
# HELPERS INTERNOS
# =============================================================================

def _guardar(save_path, nombre_archivo):
    """Guarda la figura activa en save_path/nombre_archivo si se indicó ruta."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ruta = os.path.join(save_path, nombre_archivo)
        plt.savefig(ruta, bbox_inches="tight", dpi=150)
        print(f"  Guardado: {ruta}")