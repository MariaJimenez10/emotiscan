
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
MODELO = Path("modelo/resnet50_emociones_limpio.keras")
CARPETA_TEST = Path("dataset_limpio/test")
TAMANO = (224, 224)
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("Modelo:", MODELO.resolve())
print("Existe:", MODELO.exists())

if not MODELO.exists():
    raise FileNotFoundError(MODELO)

modelo = tf.keras.models.load_model(MODELO)

print("Salida del modelo:", modelo.output_shape)
print("Orden de clases:", CLASES)

# Cargar las mismas imágenes con el generador de Keras
generador = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    str(CARPETA_TEST),
    classes=CLASES,
    target_size=TAMANO,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)

generador.reset()
prob_batch = modelo.predict(generador, verbose=0)
y_real = generador.classes
y_pred_batch = np.argmax(prob_batch, axis=1)

# Evaluar las imágenes de Enojo una por una con PIL
carpeta_enojo = CARPETA_TEST / "Enojo"
archivos = sorted(
    p for p in carpeta_enojo.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONES
)

resultados_individuales = []

for ruta in archivos:
    with Image.open(ruta) as original:
        imagen = original.convert("RGB").resize(
        TAMANO, resample=Image.Resampling.NEAREST
        )
        arreglo = np.asarray(imagen, dtype=np.float32)[None, ...]
        arreglo = preprocess_input(arreglo)

    probabilidades = modelo.predict(arreglo, verbose=0)[0]
    resultados_individuales.append(
        (ruta.name, int(np.argmax(probabilidades)),
         float(probabilidades[0]))
    )

# Comparar por nombre de archivo, no solo por porcentajes
indices_enojo = [
    i for i, ruta in enumerate(generador.filepaths)
    if Path(ruta).parent.name == "Enojo"
]

predicciones_batch = {
    Path(generador.filepaths[i]).name: int(y_pred_batch[i])
    for i in indices_enojo
}

coinciden = 0
comparados = 0

for nombre, pred_individual, prob_enojo in resultados_individuales:
    if nombre in predicciones_batch:
        comparados += 1
        if predicciones_batch[nombre] == pred_individual:
            coinciden += 1

aciertos_batch = sum(
    y_pred_batch[i] == 0 for i in indices_enojo
)

aciertos_individual = sum(
    pred == 0 for _, pred, _ in resultados_individuales
)

print("\n--- COMPARACIÓN ---")
print("Imágenes de Enojo:", len(archivos))
print("Predichas como Enojo por Keras:", aciertos_batch)
print("Predichas como Enojo individualmente:", aciertos_individual)
print("Archivos comparados:", comparados)
print("Predicción idéntica en ambos métodos:", coinciden)

print("\n--- PRIMERAS 20 IMÁGENES ---")
for nombre, pred, prob in resultados_individuales[:20]:
    print(
        f"{nombre} | Predicción: {CLASES[pred]} "
        f"| Probabilidad Enojo: {prob * 100:.2f}%"
    )
