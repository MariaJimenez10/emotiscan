import os
import cv2
import numpy as np
import logging

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Ruta del modelo
# ===========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelo_resnet50_emociones.h5"
)

logger.info("Cargando modelo...")

modelo = load_model(MODEL_PATH)

logger.info("Modelo cargado correctamente.")

# ===========================
# Emociones
# ===========================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ===========================
# Predicción
# ===========================

def predecir(rostro):

    try:

        logger.info("Paso 1: Recibí el rostro")

        rostro = cv2.resize(rostro, (224,224))

        logger.info("Paso 2: Resize correcto")

        rostro = rostro.astype(np.float32)

        logger.info("Paso 3: float32")

        rostro = preprocess_input(rostro)

        logger.info("Paso 4: preprocess_input")

        rostro = np.expand_dims(rostro, axis=0)

        logger.info("Paso 5: expand_dims")

        pred = modelo.predict(rostro, verbose=0)

        logger.info("Paso 6: model.predict OK")

        pred = pred[0]

        indice = np.argmax(pred)

        emocion = EMOCIONES[indice]

        confianza = float(pred[indice])

        logger.info(f"Predicción: {emocion} ({confianza})")

        return emocion, confianza, pred.tolist()

    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return "Neutral",0,None
    