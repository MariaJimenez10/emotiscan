import os
import gc
import cv2
import numpy as np
import logging
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# CONFIGURACIÓN TENSORFLOW
# ======================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ======================================
# RUTA DEL MODELO
# ======================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelo_resnet50_emociones.h5"
)

logger.info("Cargando modelo...")

modelo = load_model(MODEL_PATH, compile=False)

logger.info("Modelo cargado correctamente.")

# ======================================
# EMOCIONES
# ======================================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# ======================================
# PREDICCIÓN
# ======================================

def predecir(rostro):

    try:

        if rostro is None:
            return "Neutral", 0.0, None

        rostro = cv2.resize(rostro, (224, 224))

        rostro = rostro.astype(np.float32)

        rostro = preprocess_input(rostro)

        rostro = np.expand_dims(rostro, axis=0)

        pred = modelo.predict(rostro, verbose=0)[0]

        indice = np.argmax(pred)

        emocion = EMOCIONES[indice]

        confianza = float(pred[indice])

        # liberar memoria temporal
        del rostro

        gc.collect()

        return emocion, confianza, pred.tolist()

    except Exception as e:

        logger.exception(e)

        gc.collect()

        return "Neutral", 0.0, None