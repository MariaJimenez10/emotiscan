import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

MODELO = r"C:\Users\Equipo\OneDrive\Desktop\111\emotiscan\modelo_resnet50_emociones.tflite"

IMAGEN = r"C:\Users\Equipo\OneDrive\Desktop\111\emotiscan\debug_rostros\felicidad_00000.jpg"


CLASES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# CARGAR IMAGEN
# ==========================================================

img = cv2.imread(IMAGEN)

if img is None:
    print("❌ No se pudo cargar la imagen")
    exit()

print("======================================")
print("PRUEBA DIRECTA - FELICIDAD")
print("======================================")

print("Imagen original:", img.shape)


# ==========================================================
# CONVERTIR A RGB
# ==========================================================

img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


# ==========================================================
# REDIMENSIONAR
# ==========================================================

img = cv2.resize(
    img,
    (224, 224)
)

print("Imagen redimensionada:", img.shape)


# ==========================================================
# PREPROCESS INPUT
# ==========================================================

img = img.astype(np.float32)

img = preprocess_input(img)

img = np.expand_dims(
    img,
    axis=0
)

print("Tensor final:", img.shape)
print("Tipo:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())


# ==========================================================
# CARGAR TFLITE
# ==========================================================

print("\n======================================")
print("CARGANDO MODELO")
print("======================================")

interpreter = tf.lite.Interpreter(
    model_path=MODELO
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("Entrada:", input_details[0]["shape"])
print("Tipo entrada:", input_details[0]["dtype"])

print("Salida:", output_details[0]["shape"])
print("Tipo salida:", output_details[0]["dtype"])


# ==========================================================
# EJECUTAR MODELO
# ==========================================================

interpreter.set_tensor(
    input_details[0]["index"],
    img
)

interpreter.invoke()


# ==========================================================
# OBTENER RESULTADO
# ==========================================================

resultado = interpreter.get_tensor(
    output_details[0]["index"]
)[0]


print("\n======================================")
print("SALIDA DEL MODELO")
print("======================================")

print(resultado)

print("\nSuma:", resultado.sum())


# ==========================================================
# MOSTRAR PROBABILIDADES
# ==========================================================

print("\n======================================")
print("PROBABILIDADES")
print("======================================")


for clase, probabilidad in zip(
    CLASES,
    resultado
):

    print(
        f"{clase}: {probabilidad * 100:.2f}%"
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

indice = np.argmax(resultado)

emocion = CLASES[indice]

confianza = resultado[indice] * 100


print("\n======================================")
print("RESULTADO FINAL")
print("======================================")

print(
    f"🏆 Predicción: {emocion}"
)

print(
    f"🎯 Confianza: {confianza:.2f}%"
)

print("======================================")