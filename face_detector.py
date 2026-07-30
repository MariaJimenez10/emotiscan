import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el clasificador una sola vez
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detectar_rostro(img):
    """Detecta el rostro y devuelve la imagen recortada en tamaño 224x224"""

    try:
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(caras) == 0:
            logger.warning("No se detectó ningún rostro.")
            return None

        # Tomar el rostro más grande
        x, y, w, h = max(caras, key=lambda r: r[2] * r[3])

        rostro = img[y:y+h, x:x+w]

        rostro = cv2.resize(rostro, (224, 224))

        logger.info("✅ Rostro detectado correctamente")

        return rostro

    except Exception as e:
        logger.error(f"Error detectando rostro: {e}")
        return None