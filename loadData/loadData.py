# realizado por Mariana Valderrama

import pandas as pd
from Limpieza import limpiar_datos

# =========================
# CARGA DEL DATASET (CSV)
# =========================
df = pd.read_csv("../dataset_reemplazabilidad_ia.csv")

print("Dataset cargado correctamente\n")

# =========================
# LIMPIEZA DE COLUMNAS
# =========================
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("\n", "")
df.columns = df.columns.str.lower()

print("Columnas después de limpieza:")
print(df.columns.tolist(), "\n")

# =========================
# ELIMINAR COLUMNAS INNECESARIAS
# =========================
# En nuestro dataset actual NO hay columnas basura (como correos o timestamps),


cols_eliminar = [
    
]

df = df.drop(columns=cols_eliminar, errors='ignore')

print("Columnas eliminadas correctamente:")
print(cols_eliminar, "\n")

# =========================
# RENOMBRAR COLUMNAS 
# =========================
# nombres más cortos:

mapeo_columnas = {
    "nivel_experiencia": "exp",
    "nivel_rutina": "rutina",
    "nivel_estructuracion": "estructuracion",
    "nivel_creatividad": "creatividad",
    "resolucion_problemas_complejos": "resolucion",
    "interaccion_humana": "interaccion",
    "porcentaje_tareas_automatizables": "automatizacion"
}

df = df.rename(columns=mapeo_columnas)

print("Columnas renombradas correctamente:")
print(df.columns.tolist(), "\n")

# =========================
# VISTA GENERAL
# =========================
# 
df = limpiar_datos(df)

print(df.head())
print("\nInformación del dataset:")
print(df.info())