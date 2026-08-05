import cv2
import logging
import numpy as np

from deepface import DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# CASCADE DE RESPALDO
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# DETECTAR ROSTRO
# ==========================================

def detectar_rostro(img):

    try:

        if img is None:
            return None

        # ==========================================
        # 1. DEEPFACE
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

                # DeepFace devuelve imagen normalizada (0-1)
                if rostro.max() <= 1:
                    rostro = (rostro * 255).astype(np.uint8)

                rostro = cv2.cvtColor(
                    rostro,
                    cv2.COLOR_RGB2BGR
                )

                rostro = cv2.resize(
                    rostro,
                    (224,224)
                )

                logger.info("✅ Rostro detectado con DeepFace")

                return rostro

        except Exception as e:

            logger.warning(
                f"DeepFace falló: {e}"
            )

        # ==========================================
        # 2. OPENCV (RESPALDO)
        # ==========================================

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        caras = face_cascade.detectMultiScale(

            gray,

            scaleFactor=1.1,

            minNeighbors=5,

            minSize=(60,60)

        )

        if len(caras) == 0:

            logger.warning(
                "No se detectó rostro."
            )

            return None

        x,y,w,h = max(
            caras,
            key=lambda r:r[2]*r[3]
        )

        rostro = img[y:y+h, x:x+w]

        rostro = cv2.resize(
            rostro,
            (224,224)
        )

        logger.info(
            "✅ Rostro detectado con OpenCV"
        )

        return rostro

    except Exception as e:

        logger.error(e)

        return None