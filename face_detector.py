import cv2
import logging
import numpy as np
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Detector OpenCV de respaldo
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# Detectar rostro
# ==========================================

def detectar_rostro(img):

    try:

        if img is None:
            return None

        # ==========================================
        # 1. Intentar con DeepFace
        # ==========================================

        try:

            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="opencv",
                enforce_detection=False,
                align=True
            )

            if len(faces) > 0:

                rostro = faces[0]["face"]

                # DeepFace devuelve valores entre 0 y 1
                if rostro.max() <= 1:
                    rostro = (rostro * 255).astype(np.uint8)

                # RGB -> BGR
                rostro = cv2.cvtColor(
                    rostro,
                    cv2.COLOR_RGB2BGR
                )

                rostro = cv2.resize(
                    rostro,
                    (224, 224)
                )

                logger.info("✅ Rostro detectado con DeepFace")

                return rostro

        except Exception as e:

            logger.warning(f"DeepFace falló: {e}")

        # ==========================================
        # 2. Respaldo con OpenCV
        # ==========================================

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(caras) == 0:

            logger.warning("No se detectó ningún rostro.")

            return None

        x, y, w, h = max(
            caras,
            key=lambda r: r[2] * r[3]
        )

        # Agregar un pequeño margen
        margen = int(0.15 * max(w, h))

        x = max(0, x - margen)
        y = max(0, y - margen)

        w = min(img.shape[1] - x, w + margen * 2)
        h = min(img.shape[0] - y, h + margen * 2)

        rostro = img[y:y+h, x:x+w]

        rostro = cv2.resize(
            rostro,
            (224, 224)
        )

        logger.info("✅ Rostro detectado con OpenCV")

        return rostro

    except Exception as e:

        logger.exception(e)

        return None