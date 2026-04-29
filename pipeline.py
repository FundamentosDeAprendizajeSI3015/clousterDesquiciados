
from loadData.loadData import cargar_datos
from loadData.Limpieza import limpiar_datos
from EDA.EDA import ejecutar_eda
from no_supervisado.no_supervisado import ejecutar_no_supervisado, reasignar_etiquetas
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = ["exp", "rutina", "estructuracion", "creatividad", "resolucion", "interaccion"]


def preparar_carpeta_visualizaciones(base_dir):
    """
    Crea la carpeta visualizaciones/no_supervisado si no existe y
    retorna su ruta absoluta para que los gráficos se guarden ahí.
    """
    ruta = os.path.join(base_dir, "visualizaciones", "no_supervisado")
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

    ruta_vis_ns = preparar_carpeta_visualizaciones(os.path.dirname(__file__))

    resultados_ns = ejecutar_no_supervisado(
        X_scaled,
        df=df,
        target_col="automatizacion_cat",
        k_min=2,
        k_max=12,
        save_path=ruta_vis_ns,
    )

    # 10. Reasignación de etiquetas + gráficas (confusión, PCA, distribución)
    if resultados_ns["mapeo_clusters"] is not None:
        etiquetas_reasignadas = reasignar_etiquetas(
            df,
            resultados_ns["etiquetas"],
            X_scaled,
            resultados_ns["mapeo_clusters"],
            target_col="automatizacion_cat",
            save_path=ruta_vis_ns,
        )

    print("\n" + "="*60)
    print("RESUMEN ANÁLISIS NO SUPERVISADO")
    print("="*60)
    print(f"  K final seleccionado: {resultados_ns['k_final']}")
    if resultados_ns["mapeo_clusters"] is not None:
        print("\n  Mapeo clusters → clase real:")
        for cluster_id, clase in resultados_ns["mapeo_clusters"].items():
            print(f"    Cluster {cluster_id} → {clase}")


if __name__ == "__main__":
    main()

