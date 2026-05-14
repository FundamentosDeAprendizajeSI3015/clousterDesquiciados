
from loadData.loadData import cargar_datos
from loadData.Limpieza import limpiar_datos
from EDA.EDA import ejecutar_eda
from no_supervisado.no_supervisado import ejecutar_no_supervisado
from indexesScore.Score import calcular_todas_metricas, imprimir_reporte
from indexesScore.score_nosuperviced.graficas import generar_todas_graficas
from SVM.SVM import ejecutar_svm
from supervised.supervised import ejecutar_supervisado
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = ["exp", "rutina", "estructuracion", "creatividad", "resolucion", "interaccion"]


def preparar_carpeta_visualizaciones(base_dir, subcarpeta):
    """Crea la subcarpeta dentro de visualizaciones/ y retorna su ruta."""
    ruta = os.path.join(base_dir, "visualizaciones", subcarpeta)
    os.makedirs(ruta, exist_ok=True)
    print(f"\n  Carpeta de visualizaciones lista: {ruta}")
    return ruta


def _cargar_splits(base_dir, persona):
    """Carga los archivos CSV de la carpeta splits/ para el modo distribuido."""
    splits_dir = os.path.join(base_dir, "splits")
    ruta_train = os.path.join(splits_dir, f"train_persona_{persona}.csv")
    ruta_val   = os.path.join(splits_dir, "validacion.csv")
    ruta_test  = os.path.join(splits_dir, "test.csv")

    for ruta in (ruta_train, ruta_val, ruta_test):
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró '{ruta}'.\n"
                "Ejecuta primero:  python preparar_splits.py"
            )

    df_train = pd.read_csv(ruta_train)
    df_val   = pd.read_csv(ruta_val)
    df_test  = pd.read_csv(ruta_test)
    return df_train, df_val, df_test


def main():

    parser = argparse.ArgumentParser(description="Pipeline de procesamiento de datos")
    parser.add_argument(
        "--persona", type=int, choices=range(1, 7), default=None,
        metavar="N",
        help="Modo distribuido: número de persona 1-6. "
             "Requiere haber ejecutado preparar_splits.py antes."
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*60)
    print("PIPELINE DE PROCESAMIENTO DE DATOS")
    if args.persona:
        print(f"  MODO DISTRIBUIDO — Persona {args.persona} / 6")
    print("="*60)

    if args.persona:
        # Modo distribuido: cargar splits pre-generados
        print(f"\nCargando splits de la carpeta splits/...")
        df_train, df_val, df_test = _cargar_splits(base_dir, args.persona)
        print(f"  train_persona_{args.persona}.csv : {len(df_train):,} filas")
        print(f"  validacion.csv                  : {len(df_val):,} filas")
        print(f"  test.csv                        : {len(df_test):,} filas")
        df = df_train  # EDA y clustering solo sobre datos de entrenamiento
        df_val_arg  = df_val
        df_test_arg = df_test
    else:
        # Modo normal: cargar y limpiar el dataset completo
        df = cargar_datos()
        df = limpiar_datos(df)

        # Guardar dataset limpio
        ruta_salida = os.path.join(base_dir, "dataset_limpio.csv")
        df.to_csv(ruta_salida, index=False)
        print("\nDataset limpio guardado")
        print(f" Archivo generado: {ruta_salida}")
        df_val_arg  = None
        df_test_arg = None

    # 4. EDA
    print("\nIniciando Análisis Exploratorio de Datos...")
    resultados_eda = ejecutar_eda(df, target_col="automatizacion")
    print(resultados_eda["stats"])

    # 9. Análisis No Supervisado — K-Means con método del codo + Silhouette
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ruta_vis_ns = preparar_carpeta_visualizaciones(base_dir, "no_supervisado")

    resultados_ns = ejecutar_no_supervisado(
        X_scaled,
        df=df,
        target_col="automatizacion_cat",
        k_min=2,
        k_max=12,
        save_path=ruta_vis_ns,
    )

    # 10. Corrección de etiquetas — resumen (la corrección ocurre dentro de ejecutar_no_supervisado)
    print("\n" + "="*60)
    print("RESUMEN ANÁLISIS NO SUPERVISADO")
    print("="*60)
    print(f"  K final seleccionado: {resultados_ns['k_final']}")
    if resultados_ns["mapeo_clusters"] is not None:
        print("\n  Mapeo clusters → clase real:")
        for cluster_id, clase in resultados_ns["mapeo_clusters"].items():
            print(f"    Cluster {cluster_id} → {clase}")

    # 11. Métricas del clustering (Accuracy, F1, Precisión, Recall, R²)
    if resultados_ns["etiquetas_corregidas"] is not None:
        resultados_metricas = calcular_todas_metricas(
            y_true_cat           = df["automatizacion_cat"].values,
            y_continuo           = df["automatizacion"].values,
            etiquetas_raw        = resultados_ns["etiquetas"],
            etiquetas_corregidas = resultados_ns["etiquetas_corregidas"],
        )
        imprimir_reporte(resultados_metricas)

        ruta_score = os.path.join(base_dir, "visualizaciones", "no_supervisado_metricas")
        generar_todas_graficas(resultados_metricas, save_path=ruta_score)

    # -----------------------------------------------------------------------------
    # 12. MODELO SUPERVISADO — SVM
    # -----------------------------------------------------------------------------

    ejecutar_svm(
        df,
        base_dir=base_dir,
        val_df=df_val_arg,
        test_df=df_test_arg,
    )

    # -----------------------------------------------------------------------------
    # 13. MODELOS SUPERVISADOS — Regresión Logística, Árbol, Random Forest
    # -----------------------------------------------------------------------------

    ruta_vis_sup = preparar_carpeta_visualizaciones(base_dir, "supervisado")

    ejecutar_supervisado(
        df,
        target_col="automatizacion_cat",
        task="classification",
        features=FEATURES,
        save_path=ruta_vis_sup,
        val_df=df_val_arg,
        test_df=df_test_arg,
    )


if __name__ == "__main__":
    main()

