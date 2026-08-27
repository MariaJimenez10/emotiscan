import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

MODEL_PATH = "modelo_resnet50_emociones.h5"

DATASET_PATH = "dataset/train"

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# CARGAR MODELO
# ==========================================================

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

        continue


    archivo = archivos[0]

    ruta = os.path.join(
        carpeta,
        archivo
    )


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
        verbose=0
    )[0]


    print("\nPredicción:")

    for i, valor in enumerate(pred):

        print(
            f"{EMOCIONES[i]}: "
            f"{valor:.6f} "
            f"({valor * 100:.2f}%)"
        )


    indice = int(
        np.argmax(pred)
    )


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