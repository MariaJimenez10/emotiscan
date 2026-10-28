
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)
from tensorflow.keras.applications.resnet50 import preprocess_input

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
CARPETA = Path("dataset_limpio/test/Enojo")
CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]

if not MODELO.exists():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not CARPETA.exists():
    raise FileNotFoundError(f"No existe la carpeta: {CARPETA}")

archivos = sorted(
    p for p in CARPETA.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

if not archivos:
    raise RuntimeError("No se encontraron imágenes de Enojo.")

modelo = tf.keras.models.load_model(MODELO)

print("Modelo experimental cargado.")
print("Clases:", CLASES)
print("Se analizarán hasta 20 imágenes de Enojo.\n")

for ruta in archivos[:20]:
    imagen = load_img(ruta, target_size=(224, 224), color_mode="rgb")
    arreglo = img_to_array(imagen).astype(np.float32)
    arreglo = np.expand_dims(arreglo, axis=0)
    arreglo = preprocess_input(arreglo)

    probabilidades = modelo.predict(arreglo, verbose=0)[0]
    orden = np.argsort(probabilidades)[::-1]

    print(f"\nImagen: {ruta.name}")
    for indice in orden:
        print(f"  {CLASES[indice]}: {probabilidades[indice] * 100:.2f}%")
