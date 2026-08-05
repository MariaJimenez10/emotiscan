import os
import cv2
import gc
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Cargar modelo TFLite
# ==========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelo_resnet50_emociones.tflite"
)

logger.info("Cargando modelo TFLite...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info("Modelo TFLite cargado correctamente.")

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

def predecir(rostro):

    try:

        rostro = cv2.resize(rostro, (224,224))
        rostro = rostro.astype(np.float32)
        rostro = preprocess_input(rostro)
        rostro = np.expand_dims(rostro, axis=0)

        interpreter.set_tensor(
            input_details[0]["index"],
            rostro
        )

        interpreter.invoke()

        pred = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        indice = np.argmax(pred)

        emocion = EMOCIONES[indice]

        confianza = float(pred[indice])

        del rostro
        gc.collect()

        return emocion, confianza, pred.tolist()

    except Exception as e:

        logger.exception(e)

        return "Neutral", 0.0, None