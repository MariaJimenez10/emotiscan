import os
import random
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELO = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.keras"
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

IMG_SIZE = 224


# ==========================================================
# CARGAR MODELO
# ==========================================================

print("=" * 60)
print("🔬 DIAGNÓSTICO DEL MODELO")
print("=" * 60)

print("\n🧠 Cargando modelo...")

model = load_model(MODELO)

print("✅ Modelo cargado")


# ==========================================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================================

def predecir_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        raise ValueError(
            f"No se pudo leer: {ruta}"
        )

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

    pred = model.predict(
        imagen,
        verbose=0
    )[0]

    indice = int(
        np.argmax(pred)
    )

    return pred, EMOCIONES[indice]


# ==========================================================
# BUSCAR UNA IMAGEN DE CADA CLASE
# ==========================================================

print("\n" + "=" * 60)
print("📂 BUSCANDO IMÁGENES DEL TEST")
print("=" * 60)

for emocion in EMOCIONES:

    carpeta = os.path.join(
        TEST_DIR,
        emocion
    )

    archivos = []

    if os.path.exists(carpeta):

        for archivo in os.listdir(carpeta):

            if archivo.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                archivos.append(
                    os.path.join(
                        carpeta,
                        archivo
                    )
                )

    if not archivos:

        print(
            f"\n❌ No hay imágenes para {emocion}"
        )

        continue

    # Tomamos hasta 5 imágenes aleatorias
    cantidad = min(
        5,
        len(archivos)
    )

    seleccionadas = random.sample(
        archivos,
        cantidad
    )

    print("\n" + "-" * 60)
    print(
        f"🎭 CLASE REAL: {emocion}"
    )
    print("-" * 60)

    correctas = 0

    for ruta in seleccionadas:

        pred, principal = predecir_imagen(
            ruta
        )

        print(
            f"\n📷 {os.path.basename(ruta)}"
        )

        for i, nombre in enumerate(
            EMOCIONES
        ):

            print(
                f"   {nombre:<12}: "
                f"{pred[i] * 100:7.2f}%"
            )

        print(
            f"   🏆 Predicción: {principal}"
        )

        if principal == emocion:

            correctas += 1

    print(
        f"\n📊 Resultado: "
        f"{correctas}/{cantidad} correctas"
    )


# ==========================================================
# PROBAR EL ROSTRO DE LA CÁMARA
# ==========================================================

ROSTRO_CAMARA = os.path.join(
    BASE_DIR,
    "debug_rostros",
    "rostro_detectado.jpg"
)

print("\n" + "=" * 60)
print("📷 ROSTRO DE LA CÁMARA")
print("=" * 60)

if os.path.exists(ROSTRO_CAMARA):

    pred, principal = predecir_imagen(
        ROSTRO_CAMARA
    )

    for i, nombre in enumerate(
        EMOCIONES
    ):

        print(
            f"{nombre:<12}: "
            f"{pred[i] * 100:.6f}%"
        )

    print(
        f"\n🏆 Predicción cámara: "
        f"{principal}"
    )

else:

    print(
        "❌ No existe rostro_detectado.jpg"
    )


print("\n" + "=" * 60)
print("🏁 DIAGNÓSTICO TERMINADO")
print("=" * 60)