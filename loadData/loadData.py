# loadData.py

import pandas as pd
import os

def cargar_datos():
    
    # construir la ruta al archivo csv ubicado un nivel arriba
    ruta = os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset_reemplazabilidad_sintetico.csv"
    )

    # cargar el dataset desde el archivo csv
    df = pd.read_csv(ruta)

    # limpiar nombres de columnas:
    # eliminar espacios, saltos de línea y convertir a minúsculas
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\n", "")
    df.columns = df.columns.str.lower()

    # renombrar columnas para facilitar el manejo en el análisis
    df = df.rename(columns={
        "nivel_experiencia": "exp",
        "nivel_rutina": "rutina",
        "nivel_estructuracion": "estructuracion",
        "nivel_creatividad": "creatividad",
        "resolucion_problemas_complejos": "resolucion",
        "interaccion_humana": "interaccion",
        "porcentaje_tareas_automatizables": "automatizacion"
    })

    # confirmar que la carga y preprocesamiento básico fueron exitosos
    print("datos cargados correctamente")

    # retornar el dataframe listo 
    return df