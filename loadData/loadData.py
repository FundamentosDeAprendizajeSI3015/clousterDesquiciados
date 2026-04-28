# loadData.py
import pandas as pd
import os

def cargar_datos():
    
    ruta = os.path.join(
        os.path.dirname(__file__),
        "..",
        "dataset_reemplazabilidad_ia.csv"
    )

    df = pd.read_csv(ruta)

    # limpieza básica
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\n", "")
    df.columns = df.columns.str.lower()

    df = df.rename(columns={
        "nivel_experiencia": "exp",
        "nivel_rutina": "rutina",
        "nivel_estructuracion": "estructuracion",
        "nivel_creatividad": "creatividad",
        "resolucion_problemas_complejos": "resolucion",
        "interaccion_humana": "interaccion",
        "porcentaje_tareas_automatizables": "automatizacion"
    })

    print("✔ Datos cargados correctamente")
    
    return df