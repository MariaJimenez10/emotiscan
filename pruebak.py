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

MODELO_KERAS = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.keras"
)

MODELO_TFLITE = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)

RUTA_IMAGEN = os.path.join(
    BASE_DIR,
    "debug_rostros",
    "rostro_detectado.jpg"
)

IMG_SIZE = 224

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# COMPROBAR ARCHIVOS
# ==========================================================

print("=" * 70)
print("🧪 COMPARACIÓN KERAS vs TFLITE")
print("=" * 70)

if not os.path.exists(MODELO_KERAS):
    raise FileNotFoundError(
        f"No existe:\n{MODELO_KERAS}"
    )

if not os.path.exists(MODELO_TFLITE):
    raise FileNotFoundError(
        f"No existe:\n{MODELO_TFLITE}"
    )

if not os.path.exists(RUTA_IMAGEN):
    raise FileNotFoundError(
        f"No existe:\n{RUTA_IMAGEN}"
    )


# ==========================================================
# CARGAR IMAGEN
# ==========================================================

imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    raise ValueError(
        f"No se pudo abrir:\n{RUTA_IMAGEN}"
    )

print()
print(f"📷 Imagen: {RUTA_IMAGEN}")
print(f"📐 Tamaño original: {imagen.shape}")


# ==========================================================
# PREPROCESAMIENTO
# ==========================================================

imagen_rgb = cv2.cvtColor(
    imagen,
    cv2.COLOR_BGR2RGB
)

imagen_rgb = cv2.resize(
    imagen_rgb,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_AREA
)

imagen_rgb = imagen_rgb.astype(
    np.float32
)

imagen_rgb = preprocess_input(
    imagen_rgb
)

imagen_rgb = np.expand_dims(
    imagen_rgb,
    axis=0
)

print(
    f"📦 Entrada preparada: {imagen_rgb.shape}"
)

print(
    f"📊 Rango: "
    f"min={imagen_rgb.min():.2f}, "
    f"max={imagen_rgb.max():.2f}"
)


# ==========================================================
# FUNCIÓN PARA MOSTRAR RESULTADOS
# ==========================================================

def mostrar_resultado(nombre_modelo, probabilidades):

    probabilidades = np.asarray(
        probabilidades,
        dtype=np.float32
    )

    probabilidades = np.squeeze(
        probabilidades
    )

    print()
    print("=" * 70)
    print(f"🎭 {nombre_modelo}")
    print("=" * 70)

    print(
        f"📊 Suma: {np.sum(probabilidades):.6f}"
    )

    for i, emocion in enumerate(EMOCIONES):

        print(
            f"{emocion:<12}: "
            f"{probabilidades[i] * 100:>8.4f}%"
        )

    indice = int(
        np.argmax(probabilidades)
    )

    print()
    print(
        f"🏆 PREDICCIÓN: "
        f"{EMOCIONES[indice]}"
    )

    print(
        f"🎯 CONFIANZA: "
        f"{probabilidades[indice] * 100:.4f}%"
    )

    print("=" * 70)

    return EMOCIONES[indice], probabilidades


# ==========================================================
# 1. MODELO KERAS
# ==========================================================

print()
print("=" * 70)
print("🧠 CARGANDO MODELO KERAS")
print("=" * 70)

modelo = load_model(
    MODELO_KERAS
)

print("✅ Modelo Keras cargado")


salida_keras = modelo.predict(
    imagen_rgb,
    verbose=0
)

emocion_keras, probabilidades_keras = mostrar_resultado(
    "MODELO KERAS",
    salida_keras
)


# ==========================================================
# 2. MODELO TFLITE
# ==========================================================

print()
print("=" * 70)
print("📦 CARGANDO MODELO TFLITE")
print("=" * 70)

interpreter = tf.lite.Interpreter(
    model_path=MODELO_TFLITE
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Modelo TFLite cargado")

print(
    f"📥 Entrada: "
    f"{input_details[0]['shape']}"
)

print(
    f"📤 Salida: "
    f"{output_details[0]['shape']}"
)

interpreter.set_tensor(
    input_details[0]["index"],
    imagen_rgb.astype(np.float32)
)

interpreter.invoke()

salida_tflite = interpreter.get_tensor(
    output_details[0]["index"]
)

emocion_tflite, probabilidades_tflite = mostrar_resultado(
    "MODELO TFLITE",
    salida_tflite
)


# ==========================================================
# COMPARACIÓN
# ==========================================================

print()
print("=" * 70)
print("🔬 COMPARACIÓN FINAL")
print("=" * 70)

print(
    f"Keras : {emocion_keras}"
)

print(
    f"TFLite: {emocion_tflite}"
)

print()

print("Diferencias entre probabilidades:")

for i, emocion in enumerate(EMOCIONES):

    keras_pct = probabilidades_keras[i] * 100
    tflite_pct = probabilidades_tflite[i] * 100

    diferencia = abs(
        keras_pct - tflite_pct
    )

    print(
        f"{emocion:<12}: "
        f"{diferencia:.6f}%"
    )


print()
print("=" * 70)

if emocion_keras == emocion_tflite:

    print(
        "✅ KERAS Y TFLITE COINCIDEN"
    )

    print(
        "➡️ La conversión a TFLite NO parece ser el problema."
    )

else:

    print(
        "⚠️ KERAS Y TFLITE NO COINCIDEN"
    )

    print(
        "➡️ Hay que revisar la conversión/preprocesamiento."
    )

print("=" * 70)