# realizado por Mariana Valderrama

import pandas as pd
import os
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


df = limpiar_datos(df)

# =========================
# GUARDAR DATASET LIMPIO
# =========================
print("\nGuardando dataset limpio...")

ruta_salida = os.path.join(
    os.path.dirname(__file__),  # carpeta actual (loadData)
    "..",                       # subir un nivel
    "dataset_limpio.csv"        # nombre del nuevo archivo
)

df.to_csv(ruta_salida, index=False)

print(f"✔ Dataset limpio guardado en: {ruta_salida}")

# =========================
# VISTA GENERAL
# =========================
# 
print(df.head())
print("\nInformación del dataset:")
print(df.info())