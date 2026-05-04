
from loadData.loadData import cargar_datos
from loadData.Limpieza import limpiar_datos
from EDA.EDA import ejecutar_eda
from no_supervisado.no_supervisado import ejecutar_no_supervisado
from indexesScore.Score import calcular_todas_metricas, imprimir_reporte
from indexesScore.score_nosuperviced.graficas import generar_todas_graficas
from SVM.SVM import ejecutar_svm
from supervised.supervised import ejecutar_supervisado
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


def main():

    print("\n" + "="*60)
    print("PIPELINE DE PROCESAMIENTO DE DATOS")
    print("="*60)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Limpiar datos
    df = limpiar_datos(df)

    # 3. Guardar dataset limpio
    ruta_salida = os.path.join(
        os.path.dirname(__file__),
        "dataset_limpio.csv"
    )
    df.to_csv(ruta_salida, index=False)
    print("\nPipeline ejecutado correctamente")
    print(f" Archivo generado: {ruta_salida}")

    # 4. EDA
    print("\nIniciando Análisis Exploratorio de Datos...")
    resultados_eda = ejecutar_eda(df, target_col="automatizacion")
    print(resultados_eda["stats"])

    # 9. Análisis No Supervisado — K-Means con método del codo + Silhouette
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ruta_vis_ns = preparar_carpeta_visualizaciones(os.path.dirname(__file__), "no_supervisado")

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

        ruta_score = os.path.join(
            os.path.dirname(__file__),
            "visualizaciones", "no_supervisado_metricas"
        )
        generar_todas_graficas(resultados_metricas, save_path=ruta_score)

    # -----------------------------------------------------------------------------
    # 12. MODELO SUPERVISADO — SVM
    # -----------------------------------------------------------------------------

    ejecutar_svm(
        df,
        base_dir=os.path.dirname(__file__)
    )

    # -----------------------------------------------------------------------------
    # 13. MODELOS SUPERVISADOS — Regresión Logística, Árbol, Random Forest
    # -----------------------------------------------------------------------------

    ruta_vis_sup = preparar_carpeta_visualizaciones(os.path.dirname(__file__), "supervisado")

    ejecutar_supervisado(
        df,
        target_col="automatizacion_cat",
        task="classification",
        features=FEATURES,
        save_path=ruta_vis_sup,
    )


if __name__ == "__main__":
    main()

