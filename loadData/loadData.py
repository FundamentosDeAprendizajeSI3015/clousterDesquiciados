#realizado por Mariana Valderrama
# CARGA DEL DATA SET
# Aquí cargamos el archivo de Excel que contiene las respuestas de la encuesta.
df = pd.read_excel("")

# Limpieza de nombres de columnas
# Antes de trabajar con las variables, limpiamos sus nombres.
# Esto evita errores cuando accedemos a ellas más adelante.
df.columns = df.columns.str.strip()        # Elimina espacios al inicio o al final
df.columns = df.columns.str.replace("\n", "")  # Quita saltos de línea ocultos
df.columns = df.columns.str.lower()        # Convierte todo a minúsculas para mantener uniformidad

# Eliminar columnas innecesarias
# Quitamos información como correos, nombres o marcas de tiempo, ya que no aportan valor al análisis estadístico.
cols_eliminar = [
    '',
    '',
    '',
    '',
    '',
    ''
]

df = df.drop(columns=cols_eliminar, errors='ignore')  # Si alguna columna no existe, simplemente la ignora

print("\nColumnas eliminadas correctamente:")
print(cols_eliminar)

# RENOMBRAR COLUMNAS LARGAS A VARIABLES MÁS MANEJABLES
# Las preguntas originales son muy largas, así que las convertimos en nombres más cortos y fáciles de usar dentro del código.
mapeo_columnas = {
    "",
    "",
    ""
}

df = df.rename(columns=mapeo_columnas)

print("\nColumnas renombradas correctamente:")
print(df.columns.tolist())
