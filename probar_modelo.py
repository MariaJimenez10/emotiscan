import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo",
    "mejor_modelo.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

IMG_SIZE = 224

CLASSES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]


# ==========================================================
# CARGAR MODELO
# ==========================================================

print("=" * 70)
print("CARGANDO MODELO")
print("=" * 70)

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Modelo cargado.")
print("Entrada:", model.input_shape)
print("Salida :", model.output_shape)

print()


# ==========================================================
# PREPARAR IMAGEN
# ==========================================================

def preparar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise ValueError(
            f"No se pudo leer: {ruta}"
        )

    # OpenCV = BGR
    # entrenamiento = RGB
    imagen = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB
    )

    imagen = cv2.resize(
        imagen,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    imagen = imagen.astype(
        np.float32
    )

    imagen = preprocess_input(
        imagen
    )

    imagen = np.expand_dims(
        imagen,
        axis=0
    )

    return imagen


# ==========================================================
# BUSCAR UNA IMAGEN DE CADA CLASE
# ==========================================================

print("=" * 70)
print("PROBANDO UNA IMAGEN REAL DEL DATASET POR CLASE")
print("=" * 70)


for clase_real in CLASSES:

    carpeta = os.path.join(
        TEST_DIR,
        clase_real
    )

    archivos = [
        archivo
        for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        )
    ]

    if not archivos:

        print()
        print(
            f"No existen imágenes para {clase_real}"
        )

        continue


    # Primera imagen
    archivo = archivos[0]

    ruta = os.path.join(
        carpeta,
        archivo
    )


    imagen = preparar_imagen(
        ruta
    )


    pred = model.predict(
        imagen,
        verbose=0
    )[0]


    indice = int(
        np.argmax(pred)
    )


    clase_predicha = CLASSES[
        indice
    ]


    print()
    print("-" * 70)

    print(
        f"REAL      : {clase_real}"
    )

    print(
        f"PREDICCIÓN: {clase_predicha}"
    )

    print(
        f"ARCHIVO   : {archivo}"
    )

    print()

    for nombre, prob in zip(
        CLASSES,
        pred
    ):

        print(
            f"{nombre:12s}: "
            f"{float(prob) * 100:6.2f}%"
        )


print()
print("=" * 70)
print("PRUEBA TERMINADA")
print("=" * 70)