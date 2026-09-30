import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

<<<<<<< HEAD
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
=======
MODEL_PATH = "modelo_resnet50_emociones.h5"

DATASET_PATH = "dataset/train"

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
]


# ==========================================================
# CARGAR MODELO
# ==========================================================

<<<<<<< HEAD
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
=======
print("======================================")
print("CARGANDO MODELO")
print("======================================")

model = load_model(MODEL_PATH)

print("Modelo cargado correctamente")


# ==========================================================
# PROBAR UNA IMAGEN DE CADA CLASE
# ==========================================================

for clase_idx, emocion in enumerate(EMOCIONES):

    carpeta = os.path.join(
        DATASET_PATH,
        emocion
    )

    if not os.path.exists(carpeta):

        print(f"\n❌ No existe: {carpeta}")

        continue


    archivos = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    if len(archivos) == 0:

        print(f"\n❌ No hay imágenes en {emocion}")
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        continue


<<<<<<< HEAD
    # Primera imagen
=======
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    archivo = archivos[0]

    ruta = os.path.join(
        carpeta,
        archivo
    )


<<<<<<< HEAD
    imagen = preparar_imagen(
        ruta
    )


    pred = model.predict(
        imagen,
=======
    print("\n======================================")
    print(f"CLASE REAL: {emocion}")
    print(f"IMAGEN: {archivo}")
    print("======================================")


    # ======================================================
    # CARGAR
    # ======================================================

    img = cv2.imread(ruta)


    if img is None:

        print("❌ No se pudo cargar")

        continue


    print("Imagen original:", img.shape)
    print("Tipo:", img.dtype)


    # ======================================================
    # RESIZE
    # ======================================================

    img = cv2.resize(
        img,
        (224, 224)
    )


    # ======================================================
    # FLOAT32
    # ======================================================

    img = img.astype(
        np.float32
    )


    # ======================================================
    # PREPROCESS
    # ======================================================

    img = preprocess_input(img)


    # ======================================================
    # BATCH
    # ======================================================

    img = np.expand_dims(
        img,
        axis=0
    )


    print(
        "Entrada modelo:",
        img.shape
    )


    # ======================================================
    # PREDICCIÓN
    # ======================================================

    pred = model.predict(
        img,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        verbose=0
    )[0]


<<<<<<< HEAD
=======
    print("\nPredicción:")

    for i, valor in enumerate(pred):

        print(
            f"{EMOCIONES[i]}: "
            f"{valor:.6f} "
            f"({valor * 100:.2f}%)"
        )


>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    indice = int(
        np.argmax(pred)
    )


<<<<<<< HEAD
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
=======
    print("\n--------------------------------------")

    print(
        "REAL:",
        emocion
    )

    print(
        "PREDICCIÓN:",
        EMOCIONES[indice]
    )

    print(
        "CONFIANZA:",
        f"{pred[indice] * 100:.2f}%"
    )

    print("--------------------------------------")
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
