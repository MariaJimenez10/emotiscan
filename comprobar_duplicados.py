
import os
import json
import random
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS = os.path.join(
    BASE_DIR, "modelo", "resnet50_emociones_mejorado.keras"
)
MODEL_TFLITE = os.path.join(
    BASE_DIR, "modelo", "modelo_resnet50_emociones.tflite"
)
DATASET = os.path.join(BASE_DIR, "dataset", "test")
CLASSES_FILE = os.path.join(BASE_DIR, "modelo", "clases.json")

IMG_SIZE = 224
SEED = 42
IMAGES_PER_CLASS = 5

with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

print("Clases:", CLASSES)

for path in (MODEL_KERAS, MODEL_TFLITE):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró: {path}")

print("\nCargando modelo Keras...")
keras_model = tf.keras.models.load_model(
    MODEL_KERAS, compile=False
)

print("Cargando modelo TFLite...")
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

if input_info["dtype"] != np.float32:
    raise TypeError(
        f"Este diagnóstico espera entrada float32; "
        f"el TFLite utiliza {input_info['dtype']}."
    )

if output_info["dtype"] != np.float32:
    raise TypeError(
        f"Este diagnóstico espera salida float32; "
        f"el TFLite utiliza {output_info['dtype']}."
    )

rng = random.Random(SEED)
comparaciones = []
errores = []

print("\nComparando predicciones con las mismas imágenes...")

for clase_real in CLASSES:
    carpeta = os.path.join(DATASET, clase_real)

    if not os.path.isdir(carpeta):
        print(f"AVISO: falta la carpeta {carpeta}")
        continue

    archivos = [
        os.path.join(carpeta, nombre)
        for nombre in os.listdir(carpeta)
        if nombre.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    rng.shuffle(archivos)
    seleccion = archivos[:IMAGES_PER_CLASS]

    for ruta in seleccion:
        imagen_bgr = cv2.imread(ruta)

        if imagen_bgr is None:
            errores.append(ruta)
            continue

        # Una única preparación compartida por ambos modelos.
        imagen_rgb = cv2.cvtColor(
            imagen_bgr, cv2.COLOR_BGR2RGB
        )
        imagen_rgb = cv2.resize(
            imagen_rgb, (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        lote = np.expand_dims(
            imagen_rgb.astype(np.float32), axis=0
        )
        lote = preprocess_input(lote.copy())

        # Predicción Keras.
        pred_keras = np.asarray(
            keras_model(lote, training=False)
        )[0]

        # Predicción TFLite con el mismo tensor.
        interpreter.set_tensor(input_info["index"], lote)
        interpreter.invoke()
        pred_tflite = interpreter.get_tensor(
            output_info["index"]
        )[0]

        if len(pred_keras) != len(CLASSES):
            raise ValueError("La salida Keras no coincide con clases.json.")

        if len(pred_tflite) != len(CLASSES):
            raise ValueError("La salida TFLite no coincide con clases.json.")

        if not (
            np.all(np.isfinite(pred_keras))
            and np.all(np.isfinite(pred_tflite))
        ):
            print("AVISO: salida no válida:", ruta)
            continue

        clase_k = CLASSES[int(np.argmax(pred_keras))]
        clase_t = CLASSES[int(np.argmax(pred_tflite))]

        diferencia = float(
            np.mean(np.abs(pred_keras - pred_tflite))
        )

        comparaciones.append({
            "real": clase_real,
            "keras": clase_k,
            "tflite": clase_t,
            "diferencia": diferencia,
            "archivo": os.path.basename(ruta)
        })

        print(
            f"Real: {clase_real:10} | "
            f"Keras: {clase_k:10} | "
            f"TFLite: {clase_t:10} | "
            f"Diferencia media: {diferencia:.6f} | "
            f"{os.path.basename(ruta)}"
        )

print("\n" + "=" * 65)
print("RESUMEN")
print("=" * 65)

if comparaciones:
    coincidencias = sum(
        x["keras"] == x["tflite"] for x in comparaciones
    )
    diferencia_media = np.mean([
        x["diferencia"] for x in comparaciones
    ])

    print("Imágenes comparadas:", len(comparaciones))
    print("Predicciones coincidentes:", coincidencias)
    print(
        "Porcentaje de coincidencia:",
        f"{100 * coincidencias / len(comparaciones):.2f}%"
    )
    print("Diferencia absoluta media de salidas:", 
          f"{diferencia_media:.6f}")
else:
    print("No se pudieron comparar imágenes.")

print("Imágenes no leídas:", len(errores))
print("\nNo se entrenó ni se modificó ningún modelo.")
