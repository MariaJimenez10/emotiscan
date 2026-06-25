import cv2
import numpy as np
from tensorflow.keras.models import load_model

modelo = load_model("modelo_emociones.h5")

emociones = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# MENSAJES
mensajes = {
    "Enojo": "Respira lentamente y trata de relajarte.",
    "Felicidad": "¡Qué bien! Sigue disfrutando este momento.",
    "Neutral": "Te ves tranquilo y concentrado.",
    "Tristeza": "Puedes hablar con alguien de confianza si lo necesitas."
}

imagen = cv2.imread(r"C:\Users\Equipo\OneDrive\Desktop\111\emotiscan\dataset\FER-2013\test\happy\PrivateTest_95094.jpg")
imagen = cv2.resize(imagen, (48,48))
imagen = imagen.astype("float32") / 255.0
imagen = np.expand_dims(imagen, axis=0)

prediccion = modelo.predict(imagen)

clase = np.argmax(prediccion)

emocion_detectada = emociones[clase]

print("Emoción detectada:", emocion_detectada)

# MOSTRAR MENSAJE
print("Mensaje:", mensajes[emocion_detectada])