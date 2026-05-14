# preparar_splits.py
# Ejecutar UNA SOLA VEZ antes de distribuir el trabajo entre los 6 integrantes.
# Requiere que dataset_limpio.csv ya exista (correr pipeline.py sin --persona primero).
#
# Genera en la carpeta splits/:
#   - test.csv              (20 % del total)
#   - validacion.csv        (20 % del total)
#   - train_persona_1.csv   (10 % del total)
#   - ...m   
#   - train_persona_6.csv   (10 % del total)

import os
import pandas as pd

SEED       = 42
N_PERSONAS = 6


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ruta_limpio = os.path.join(base, "dataset_limpio.csv")

    if not os.path.exists(ruta_limpio):
        raise FileNotFoundError(
            f"No se encontró '{ruta_limpio}'.\n"
            "Ejecuta primero:  python pipeline.py\n"
            "para generar el dataset limpio."
        )

    print("=" * 60)
    print("PREPARACIÓN DE SPLITS DISTRIBUIDOS")
    print("=" * 60)
    print(f"\nCargando: {ruta_limpio}")
    df = pd.read_csv(ruta_limpio)
    n  = len(df)
    print(f"Filas totales: {n:,}")

    # Shuffle uniforme con semilla fija (reproducible en todos los equipos)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print("Shuffle aplicado (random_state=42)")

    # División 60 / 20 / 20
    n_test = int(n * 0.20)
    n_val  = int(n * 0.20)

    df_test  = df.iloc[:n_test].reset_index(drop=True)
    df_val   = df.iloc[n_test : n_test + n_val].reset_index(drop=True)
    df_train = df.iloc[n_test + n_val :].reset_index(drop=True)

    print(f"\nDistribución de splits:")
    print(f"  Train total : {len(df_train):>10,} filas  ({len(df_train)/n*100:.1f}%)")
    print(f"  Validación  : {len(df_val):>10,} filas  ({len(df_val)/n*100:.1f}%)")
    print(f"  Test        : {len(df_test):>10,} filas  ({len(df_test)/n*100:.1f}%)")

    # Carpeta de salida
    carpeta = os.path.join(base, "splits")
    os.makedirs(carpeta, exist_ok=True)

    # Guardar splits compartidos
    df_val.to_csv(os.path.join(carpeta, "validacion.csv"),  index=False)
    df_test.to_csv(os.path.join(carpeta, "test.csv"),       index=False)
    print(f"\n  validacion.csv guardado  ({len(df_val):,} filas)")
    print(f"  test.csv       guardado  ({len(df_test):,} filas)")

    # Dividir train en N_PERSONAS partes iguales
    chunk = len(df_train) // N_PERSONAS
    print(f"\nDividiendo train en {N_PERSONAS} partes (~{chunk:,} filas c/u):")
    for i in range(N_PERSONAS):
        ini = i * chunk
        fin = ini + chunk if i < N_PERSONAS - 1 else len(df_train)
        parte = df_train.iloc[ini:fin].reset_index(drop=True)
        nombre = f"train_persona_{i + 1}.csv"
        parte.to_csv(os.path.join(carpeta, nombre), index=False)
        print(f"  {nombre}  →  {len(parte):,} filas  ({len(parte)/n*100:.1f}% del total)")

    print(f"\nListo. Todos los splits en: {carpeta}/")
    print("\nCada persona ejecuta:")
    print("  python pipeline.py --persona <N>   (N entre 1 y 6)")


if __name__ == "__main__":
    main()
