#realizado por Mariana Valderrama
def limpiar_datos(df,
                  columnas_a_eliminar=None,
                  columnas_importantes=None,
                  mapeo_renombrar=None,
                  columnas_ordinales=None,
                  columnas_onehot=None):
                    
    print("\n" + "="*60)
    print("LIMPIEZA DE DATOS")
    print("="*60)

    print("\n[5.1] Limpiando nombres de columnas...")
    df = limpiar_nombres_columnas(df)

    print("\n[5.2] Eliminando variables innecesarias...")
    df = eliminar_variables_innecesarias(df, columnas_a_eliminar)

    print("\n[5.3] Manejando valores nulos...")
    df = manejar_nulos(df, columnas_importantes)

    print("\n[5.4] Renombrando columnas...")
    df = renombrar_columnas(df, mapeo_renombrar)

    print("\n[5.5] Asegurando tipos numéricos...")
    df = asegurar_tipos_numericos(df)

    print("\n[5.6] Transformando variables categóricas...")
    df = transformar_categoricas(df, columnas_ordinales, columnas_onehot)

    print(f"\nLimpieza completada. Shape final: {df.shape}")
    return df
