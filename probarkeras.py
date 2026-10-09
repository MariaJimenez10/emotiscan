import os
import sys
import cv2
import numpy as np

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

IMG_SIZE = 224

CLASSES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

MODEL_PATH = os.path.join(
    "modelo",
    "resnet50_emociones_mejorado.keras"
)


# ==========================================================
# VERIFICAR ARGUMENTO
# ==========================================================

if len(sys.argv) < 2:

    print("❌ Debes indicar una imagen.")
    print("Ejemplo:")
    print("python probar_keras.py enojo.jpg")

    sys.exit(1)


imagen_path = sys.argv[1]


# ==========================================================
# VERIFICAR MODELO
# ==========================================================

if not os.path.exists(MODEL_PATH):

    print("❌ No existe el modelo:")
    print(MODEL_PATH)

    print("\nModelos .keras encontrados:")

    for root, dirs, files in os.walk("."):

        for file in files:

            if file.endswith(".keras"):

                print(
                    os.path.join(root, file)
                )

    sys.exit(1)


# ==========================================================
# VERIFICAR IMAGEN
# ==========================================================

if not os.path.exists(imagen_path):

    print(
        f"❌ No existe la imagen: {imagen_path}"
    )

    sys.exit(1)


# ==========================================================
# CARGAR MODELO
# ==========================================================

print("=" * 60)
print("🧠 CARGANDO MODELO KERAS")
print("=" * 60)

print(
    f"📁 Modelo: {MODEL_PATH}"
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Modelo Keras cargado")


# ==========================================================
# CARGAR IMAGEN
# ==========================================================

imagen = cv2.imread(
    imagen_path,
    cv2.IMREAD_COLOR
)

if imagen is None:

    print("❌ No se pudo leer la imagen.")
    sys.exit(1)


print(
    f"📷 Imagen original: {imagen.shape}"
)


# ==========================================================
# BGR → RGB
# ==========================================================

imagen = cv2.cvtColor(
    imagen,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# REDIMENSIONAR
# ==========================================================

imagen = cv2.resize(
    imagen,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_AREA
)


# ==========================================================
# FLOAT32
# ==========================================================

imagen = imagen.astype(
    np.float32
)


# ==========================================================
# PREPROCESSING DE RESNET50
# ==========================================================

imagen = preprocess_input(
    imagen
)


# ==========================================================
# BATCH
# ==========================================================

imagen = np.expand_dims(
    imagen,
    axis=0
)


print(
    f"📦 Entrada: {imagen.shape}"
)


# ==========================================================
# PREDICCIÓN
# ==========================================================

pred = model.predict(
    imagen,
    verbose=0
)[0]


print("\n" + "=" * 60)
print("📤 SALIDA RAW KERAS")
print("=" * 60)

print(pred)


# ==========================================================
# PROBABILIDADES
# ==========================================================

suma = np.sum(pred)

if (
    np.all(pred >= 0)
    and abs(suma - 1.0) < 0.05
):

    probabilidades = (
        pred / suma
    )

else:

    exp = np.exp(
        pred - np.max(pred)
    )

    probabilidades = (
        exp / np.sum(exp)
    )


# ==========================================================
# RESULTADOS
# ==========================================================

print("\n" + "=" * 60)
print("🎭 RESULTADO KERAS")
print("=" * 60)

for i, clase in enumerate(CLASSES):

    print(
        f"{clase}: "
        f"{probabilidades[i] * 100:.2f}%"
    )


indice = int(
    np.argmax(probabilidades)
)

print("-" * 60)

print(
    f"🏆 PRINCIPAL: "
    f"{CLASSES[indice]}"
)

print(
    f"📊 CONFIANZA: "
    f"{probabilidades[indice] * 100:.2f}%"
)

print("=" * 60)