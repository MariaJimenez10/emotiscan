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

        if rostro is None:
            return "Neutral", 0.0, None

        # Resize por seguridad
        rostro = cv2.resize(rostro, (224, 224))

        rostro = rostro.astype(np.float32)

        # EXACTAMENTE igual al entrenamiento
        rostro = preprocess_input(rostro)

        rostro = np.expand_dims(rostro, axis=0)

        pred = modelo.predict(
            rostro,
            verbose=0
        )[0]

        indice = np.argmax(pred)

        emocion = EMOCIONES[indice]

        confianza = float(pred[indice])

        logger.info(
            f"Predicción: {emocion} ({confianza:.2f})"
        )

        return emocion, confianza, pred.tolist()

    except Exception as e:

        logger.error(e)

        return "Neutral", 0.0, None