# Limpieza.py
import pandas as pd

def limpiar_datos(df):

    print("\n" + "="*50)
    print("INICIANDO LIMPIEZA DE DATOS")
    print("="*50)

    filas_iniciales = df.shape[0]

    # -------------------------
    # 1. Duplicados
    # -------------------------
    df = df.drop_duplicates().copy()

    # -------------------------
    # 2. Tipos numéricos
    # -------------------------
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # -------------------------
    # 3. Nulos
    # -------------------------
    # numéricos
    df = df.fillna(df.mean(numeric_only=True))

    # categóricos
    for col in df.select_dtypes(include='object').columns:
        df.loc[:, col] = df[col].fillna("desconocido")

    # -------------------------
    # 4. Escalas (1–5)
    # -------------------------
    columnas_escala = ["exp","rutina","estructuracion","creatividad","resolucion","interaccion"]

    for col in columnas_escala:
        if col in df.columns:
            df[col] = df[col].clip(1, 5)

    print("Escalas corregidas (1–5)")

    # -------------------------
    # 5. Target
    # -------------------------
    if "automatizacion" in df.columns:
        df["automatizacion"] = df["automatizacion"].clip(0, 100)
        print("Target validado (0–100)")

    # -------------------------
    # VALIDACIÓN FINAL
    # -------------------------
    filas_finales = df.shape[0]

    if df.isnull().sum().sum() == 0:
        print("\n" + "="*50)
        print(" LIMPIEZA COMPLETADA CON ÉXITO")
        print("="*50)
        print(f"Filas iniciales: {filas_iniciales}")
        print(f"Filas finales: {filas_finales}")
        print("Dataset listo para análisis \n")
    else:
        print("\n Limpieza incompleta: aún hay valores nulos")

    return df