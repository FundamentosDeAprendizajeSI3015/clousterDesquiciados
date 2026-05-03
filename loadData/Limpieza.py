# Limpieza.py
import pandas as pd

def limpiar_datos(df):

    print("\n" + "="*50)
    print("INICIANDO LIMPIEZA DE DATOS")
    print("="*50)

    filas_iniciales = df.shape[0]

    # -------------------------
    # 1. ELIMINAR DUPLICADOS
    # -------------------------
    duplicados = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    print(f"Duplicados eliminados: {duplicados}")

    # -------------------------
    # 2. DEFINIR COLUMNAS NUMÉRICAS
    # -------------------------
    columnas_numericas = [
        "exp", "rutina", "estructuracion",
        "creatividad", "resolucion", "interaccion",
        "automatizacion"
    ]

    # convertir solo esas a numérico
    for col in columnas_numericas:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    print(" Conversión a numérico realizada")

    # -------------------------
    # 3. MANEJO DE NULOS
    # -------------------------

    # numéricos → media
    df[columnas_numericas] = df[columnas_numericas].fillna(
        df[columnas_numericas].mean()
    )

    # redondear evita decimales 
    df[columnas_numericas] = df[columnas_numericas].round(0)

    # categóricos → desconocido
    for col in df.select_dtypes(include='object').columns:
        df.loc[:, col] = df[col].fillna("desconocido")

    print("Nulos tratados correctamente")

    # -------------------------
    # 4. CORREGIR ESCALAS (1–5)
    # -------------------------
    columnas_escala = [
        "exp","rutina","estructuracion",
        "creatividad","resolucion","interaccion"
    ]

    for col in columnas_escala:
        if col in df.columns:
            df.loc[:, col] = df[col].clip(1, 5)

    print(" Escalas corregidas (1–5)")

    # -------------------------
    # 5. TARGET (CATEGORIZADO)
    # -------------------------
    if "automatizacion" in df.columns:

        # asegurar rango válido
        df.loc[:, "automatizacion"] = df["automatizacion"].clip(0, 100)

        # función de categorización
        def categorizar(x):
            if x < 2:
                return "baja"
            elif x < 4:
                return "media"
            else:
                return "alta"

        df["automatizacion_cat"] = df["automatizacion"].apply(categorizar)

        print(" Target convertido a categorías (baja, media, alta)")

    # -------------------------
    # VALIDACIÓN FINAL
    # -------------------------
    filas_finales = df.shape[0]
    nulos_finales = df.isnull().sum().sum()

    if nulos_finales == 0:
        print("\n" + "="*50)
        print(" LIMPIEZA COMPLETADA CON ÉXITO")
        print("="*50)
        print(f"Filas iniciales: {filas_iniciales}")
        print(f"Filas finales: {filas_finales}")
        print("Dataset listo para análisis \n")
    else:
        print(f"\n Limpieza incompleta: aún hay {nulos_finales} valores nulos")

    return df