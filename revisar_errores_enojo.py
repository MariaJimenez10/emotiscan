from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageOps, ImageDraw
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
TAMANO = (224, 224)
BATCH_SIZE = 16

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
TEST_DIR = Path("dataset_limpio/test")
SALIDA = Path("errores_enojo_revision.jpg")
MAX_IMAGENES = 48

if not MODELO.is_file():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"No existe la carpeta: {TEST_DIR}")

print("Preparando imágenes de prueba...")

generador = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    str(TEST_DIR),
    classes=CLASES,
    target_size=TAMANO,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    interpolation="nearest"
)

print("Cargando modelo experimental...")
modelo = tf.keras.models.load_model(str(MODELO))

generador.reset()
probabilidades = modelo.predict(generador, verbose=1)

rutas = np.array(generador.filepaths)
reales = generador.classes
predichas = np.argmax(probabilidades, axis=1)

# Seleccionar imágenes cuyo rótulo real es Enojo,
# pero el modelo predijo otra emoción.
indices = [
    i for i in range(len(reales))
    if reales[i] == 0 and predichas[i] != 0
]

if not indices:
    print("No se encontraron errores de Enojo.")
    raise SystemExit(0)

# Ordenar por probabilidad de Enojo: primero los casos
# donde el modelo estuvo menos convencido de detectar Enojo.
indices.sort(key=lambda i: probabilidades[i][0])
indices = indices[:MAX_IMAGENES]


total_errores = sum(
    1
    for i in range(len(reales))
    if reales[i] == 0 and predichas[i] != 0
)

print(f"Errores de Enojo encontrados: {total_errores}")
print(f"Imágenes que se incluirán en la lámina: {len(indices)}")

columnas = 4
ancho_celda = 240
alto_celda = 220
filas = (len(indices) + columnas - 1) // columnas

lamina = Image.new(
    "RGB",
    (columnas * ancho_celda, filas * alto_celda),
    "white"
)
dibujo = ImageDraw.Draw(lamina)

for posicion, i in enumerate(indices):
    x = (posicion % columnas) * ancho_celda
    y = (posicion // columnas) * alto_celda

    ruta = Path(rutas[i])

    try:
        with Image.open(ruta) as original:
            imagen = ImageOps.exif_transpose(original).convert("RGB")
            imagen.thumbnail((210, 155))
            lamina.paste(imagen, (x + 10, y + 25))
    except Exception as error:
        dibujo.text((x + 10, y + 30), "Error al abrir imagen", fill="red")
        print(f"No se pudo abrir {ruta}: {error}")
        continue

    real = CLASES[reales[i]]
    predicha = CLASES[predichas[i]]
    p_enojo = probabilidades[i][0] * 100
    p_predicha = probabilidades[i][predichas[i]] * 100

    dibujo.text((x + 8, y + 5), f"Real: {real}", fill="black")
    dibujo.text((x + 8, y + 185), f"Predicción: {predicha}", fill="red")
    dibujo.text((x + 8, y + 200), f"Enojo: {p_enojo:.1f}% | Pred.: {p_predicha:.1f}%", fill="black")

lamina.save(SALIDA, quality=95)

print("\nLámina creada:", SALIDA.resolve())
print("No se modificaron el modelo ni la aplicación.")
