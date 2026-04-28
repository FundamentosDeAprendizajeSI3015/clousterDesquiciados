
from loadData.loadData import cargar_datos
from loadData.Limpieza import limpiar_datos
import os

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

if __name__ == "__main__":
    main()