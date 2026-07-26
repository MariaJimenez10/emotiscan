# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===============================
# Cargar el modelo una sola vez
# ===============================
MODEL_PATH = "modelo_resnet50_emociones.h5"

modelo = load_model(MODEL_PATH)

# ==========================================
# Cambia el orden según el entrenamiento
# ==========================================
EMOCIONES = [
    "Enojado",
    "Disgusto",
    "Miedo",
    "Feliz",
    "Neutral",
    "Triste",
    "Sorprendido"
]




def preparar_imagen(rostro):
    """
    Prepara la imagen para ResNet50.

    Parámetros:
        rostro: imagen BGR (OpenCV)

    Retorna:
        numpy array (1,224,224,3)
    """

    # Redimensionar
    rostro = cv2.resize(rostro, (224, 224))

    # Convertir BGR -> RGB
    rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)

    # float32
    rostro = rostro.astype(np.float32)

    # Preprocesamiento oficial ResNet50
    rostro = preprocess_input(rostro)

    # Agregar dimensión batch
    rostro = np.expand_dims(rostro, axis=0)

    print("Predicciones:", predicciones)
    print("Índice:", indice)
    print("Emoción:", emocion)
    print("Confianza:", confianza)
    
    return rostro

def predecir(rostro):

    imagen = preparar_imagen(rostro)

    print("Imagen preparada:", imagen.shape)

    predicciones = modelo.predict(imagen, verbose=0)

    print("Predicciones:", predicciones)

    indice = np.argmax(predicciones)

    print("Índice:", indice)

    emocion = EMOCIONES[indice]

    print("Emoción:", emocion)

    confianza = float(predicciones[0][indice])

    probabilidades = {
        EMOCIONES[i]: float(predicciones[0][i])
        for i in range(len(EMOCIONES))
    }

    return emocion, confianza, probabilidades
