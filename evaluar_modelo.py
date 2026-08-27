import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


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

print("✅ Modelo cargado")


# ==========================================================
# CARGAR TODAS LAS IMÁGENES
# ==========================================================

X = []
y = []


for indice, emocion in enumerate(EMOCIONES):

    carpeta = os.path.join(
        DATASET_PATH,
        emocion
    )

    if not os.path.exists(carpeta):

        print(
            f"❌ No existe: {carpeta}"
        )

        continue


    archivos = [
        f for f in os.listdir(carpeta)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    print(
        f"{emocion}: {len(archivos)} imágenes"
    )


    for archivo in archivos:

        ruta = os.path.join(
            carpeta,
            archivo
        )

        img = cv2.imread(ruta)

        if img is None:
            continue


        img = cv2.resize(
            img,
            (224, 224)
        )


        img = img.astype(
            np.float32
        )


        img = preprocess_input(
            img
        )


        X.append(img)
        y.append(indice)


# ==========================================================
# CONVERTIR
# ==========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print("\n======================================")
print("TOTAL DE IMÁGENES:", len(X))
print("======================================")


# ==========================================================
# PREDICCIÓN
# ==========================================================

print("\n🔮 Ejecutando predicciones...")

pred = model.predict(
    X,
    batch_size=32,
    verbose=1
)


y_pred = np.argmax(
    pred,
    axis=1
)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y,
    y_pred
)


print("\n======================================")
print("ACCURACY")
print("======================================")

print(
    f"{accuracy * 100:.2f}%"
)


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

cm = confusion_matrix(
    y,
    y_pred
)


print("\n======================================")
print("MATRIZ DE CONFUSIÓN")
print("======================================")

print(cm)


# ==========================================================
# REPORTE
# ==========================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y,
        y_pred,
        target_names=EMOCIONES,
        digits=4
    )
)