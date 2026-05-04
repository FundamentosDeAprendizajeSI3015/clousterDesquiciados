# =============================================================================
# MÓDULO: Análisis Exploratorio de Datos (EDA)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 3.1 INSPECCIÓN INICIAL
# =============================================================================

def inspeccion_inicial(df):
    print("\n" + "=" * 60)
    print("3.1  INSPECCIÓN INICIAL")
    print("=" * 60)

    print("\nPrimeras filas:")         # Vista rápida de los datos
    print(df.head())

    print("\nDimensiones:")            # Dimensiones (filas, columnas)
    print(df.shape)

    print("\nTipos de datos:")         # Tipos de datos
    print(df.dtypes)

    print("\nValores nulos:")         # Valores nulos por columna
    print(df.isnull().sum())

    print("\nDuplicados:")            # Duplicados
    print(df.duplicated().sum())

    return df.describe(include="all")  # Resumen estadístico completo


# =============================================================================
# 3.2 TENDENCIA CENTRAL
# =============================================================================

def tendencia_central(df):                # Cálculo de medidas centrales en variables numéricas
    print("\n" + "=" * 60)
    print("3.2  TENDENCIA CENTRAL")
    print("=" * 60)

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        print(f"\nColumna: {col}")
        print(f"Media: {df[col].mean()}")               # Promedio (sensible a outliers)
        print(f"Mediana: {df[col].median()}")           # Valor central (robusto)
        print(f"Moda: {df[col].mode().values}")         # Valor más frecuente


# =============================================================================
# 3.3 DISPERSIÓN
# =============================================================================

def dispersion(df):                         # Mide la variabilidad de los datos
    print("\n" + "=" * 60)
    print("3.3  DISPERSIÓN")
    print("=" * 60)

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        print(f"\nColumna: {col}")
        print(f"Std: {df[col].std()}")              # Dispersión promedio
        print(f"Var: {df[col].var()}")              # Dispersión promedio al cuadrado
        print(f"Rango: {df[col].max() - df[col].min()}")             # Diferencia entre máximo y mínimo
        print(f"IQR: {df[col].quantile(0.75) - df[col].quantile(0.25)}")       # Q3 - Q1 (outliers)


# =============================================================================
# 3.4 ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================

def estadisticas_descriptivas(df):         # Genera una tabla resumen con métricas estadísticas
    print("\n" + "=" * 60)
    print("3.4  TABLA RESUMEN")
    print("=" * 60)

    num_cols = df.select_dtypes(include=np.number).columns
    stats = []

    for col in num_cols:
        serie = df[col]

        stats.append({
            "variable": col,
            "media": serie.mean(),
            "mediana": serie.median(),
            "moda": serie.mode().iloc[0] if not serie.mode().empty else np.nan,
            "std": serie.std(),
            "var": serie.var(),
            "coef_var": serie.std() / serie.mean() if serie.mean() != 0 else 0,
            "min": serie.min(),
            "q1": serie.quantile(0.25),
            "q3": serie.quantile(0.75),
            "max": serie.max(),
            "iqr": serie.quantile(0.75) - serie.quantile(0.25),
            "skew": serie.skew(),               # Asimetría
            "kurtosis": serie.kurtosis()        # Concentración/extremos
        })

    return pd.DataFrame(stats)


# =============================================================================
# 3.5 VISUALIZACIONES
# =============================================================================

# Detecta outliers y resume distribución (cuartiles y mediana)
def graficar_boxplots(df, save_path):                
    num_cols = df.select_dtypes(include=np.number).columns

    plt.figure(figsize=(12, 6))
    df[num_cols].boxplot()
    plt.xticks(rotation=45)

    _guardar(save_path, "boxplots.png")
    plt.close()


# Muestra la distribución de frecuencia de cada variable
def graficar_histogramas(df, bins=10, save_path=None):
    num_cols = df.select_dtypes(include=np.number).columns

    df[num_cols].hist(bins=bins, figsize=(12, 8))

    _guardar(save_path, "histogramas.png")
    plt.close()


# Relación lineal entre variables (detección de dependencias)
def graficar_correlacion(df, save_path):
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    _guardar(save_path, "correlacion.png")
    plt.close()


# Relación entre dos variables (patrones o tendencias)
def graficar_dispersion(df, x, y, save_path):
    if x in df.columns and y in df.columns:
        plt.figure()
        sns.scatterplot(x=df[x], y=df[y])

        _guardar(save_path, f"dispersion_{x}_{y}.png")
        plt.close()


# Analiza la variable objetivo (balance o distribución)
def graficar_target(df, target_col, save_path):
    plt.figure()

    if df[target_col].nunique() < 10:
        df[target_col].value_counts().plot(kind="bar")
    else:
        sns.histplot(df[target_col], kde=True)

    _guardar(save_path, "target.png")
    plt.close()


# Visualiza relaciones entre múltiples variables
def graficar_pairplot(df, target_col, max_vars=6, save_path=None):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    cols = num_cols[:max_vars]

    sns.pairplot(df[cols + [target_col]], hue=target_col)

    _guardar(save_path, "pairplot.png")
    plt.close()

# =============================================================================
# 3.6 ANÁLISIS AVANZADO
# =============================================================================

def analisis_por_target(df, target_col):
    # Promedio de variables por nivel del target
    print("\n" + "=" * 60)
    print("ANÁLISIS POR TARGET")
    print("=" * 60)

    resumen = df.groupby(target_col).mean(numeric_only=True)

    print("\nPromedio por nivel:")
    print(resumen)

    return resumen


def correlacion_con_target(df, target_col):
    # Relación de cada variable con el target
    print("\n" + "=" * 60)
    print("CORRELACIÓN CON TARGET")
    print("=" * 60)

    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)

    print(corr)

    return corr


def detectar_outliers_iqr(df):
    # Conteo de outliers usando IQR
    print("\n" + "=" * 60)
    print("OUTLIERS (IQR)")
    print("=" * 60)

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"{col}: {len(outliers)} outliers")


def boxplots_por_target(df, target_col, save_path):
    # Boxplots separados por target
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        if col != target_col:
            plt.figure()
            sns.boxplot(x=df[target_col], y=df[col])

            _guardar(save_path, f"boxplot_{col}_vs_{target_col}.png")
            plt.close()


def histogramas_por_target(df, target_col, save_path):
    # Histogramas separados por target
    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        if col != target_col:
            plt.figure()
            sns.histplot(data=df, x=col, hue=target_col, kde=True)

            _guardar(save_path, f"hist_{col}_vs_{target_col}.png")
            plt.close()
            
            
# =============================================================================
# PIPELINE EDA
# =============================================================================

def ejecutar_eda(df, target_col):         # Ejecuta todo el flujo de análisis exploratorio
    save_path = "visualizaciones/EDA"     # Carpeta de salida

    desc = inspeccion_inicial(df)
    tendencia_central(df)
    dispersion(df)
    stats = estadisticas_descriptivas(df)
    # Análisis adicional
    analisis_por_target(df, target_col)
    correlacion_con_target(df, target_col)
    detectar_outliers_iqr(df)

    print("\nGenerando visualizaciones...")

    graficar_boxplots(df, save_path)
    graficar_histogramas(df, save_path=save_path)
    graficar_correlacion(df, save_path)
    boxplots_por_target(df, target_col, save_path)
    histogramas_por_target(df, target_col, save_path)

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        graficar_dispersion(df, num_cols[0], num_cols[1], save_path)

    graficar_target(df, target_col, save_path)
    graficar_pairplot(df, target_col, save_path=save_path)

    return {"describe": desc, "stats": stats}


# =============================================================================
# HELPER
# =============================================================================

def _guardar(save_path, nombre_archivo):       # Guarda gráficos en la ruta indicada
    os.makedirs(save_path, exist_ok=True)

    ruta = os.path.join(save_path, nombre_archivo)  # Crea carpeta si no existe

    plt.savefig(ruta, bbox_inches="tight", dpi=150)  # Guarda imagen

    print(f"Guardado: {ruta}")