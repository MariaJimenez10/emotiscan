import os
import cv2
import logging
import numpy as np


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# CARPETA DE DEBUG
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG_DIR = os.path.join(
    BASE_DIR,
    "debug_rostros"
)

os.makedirs(DEBUG_DIR, exist_ok=True)


# ==========================================================
# DETECTOR OPENCV
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    logger.error("❌ No se pudo cargar Haar Cascade")
else:
    logger.info("✅ Haar Cascade cargado correctamente")


# ==========================================================
# DEEPFACE
# ==========================================================

DeepFace = None


def cargar_deepface():

    global DeepFace

    if DeepFace is not None:
        return DeepFace

    try:

        logger.info("🧠 Cargando DeepFace...")

        from deepface import DeepFace as DF

        DeepFace = DF

        logger.info("✅ DeepFace cargado correctamente")

        return DeepFace

    except Exception as e:

        logger.exception(
            f"❌ Error cargando DeepFace: {e}"
        )

        return None


# ==========================================================
# GUARDAR IMAGEN DE DEBUG
# ==========================================================

def guardar_debug(nombre, imagen):

    try:

        if imagen is None:
            return

        if not isinstance(imagen, np.ndarray):
            return

        if imagen.size == 0:
            return

        ruta = os.path.join(
            DEBUG_DIR,
            nombre
        )

        cv2.imwrite(
            ruta,
            imagen
        )

        logger.info(
            f"📸 Debug guardado: {ruta}"
        )

    except Exception as e:

        logger.warning(
            f"⚠️ Error guardando debug: {e}"
        )


# ==========================================================
# DETECTAR ROSTRO
# ==========================================================

def detectar_rostro(img):

    logger.info(
        "🔥 detectar_rostro() FUE LLAMADA"
    )

    try:

        # ==================================================
        # VALIDAR IMAGEN
        # ==================================================

        if img is None:

            logger.error(
                "❌ La imagen es None"
            )

            return None


        if not isinstance(img, np.ndarray):

            logger.error(
                f"❌ Tipo de imagen inválido: {type(img)}"
            )

            return None


        if img.size == 0:

            logger.error(
                "❌ La imagen está vacía"
            )

            return None


        logger.info(
            "=========================================="
        )

        logger.info(
            "🔍 INICIANDO DETECCIÓN DE ROSTRO"
        )

        logger.info(
            f"📷 Imagen: {img.shape}"
        )

        logger.info(
            f"📷 Tipo: {img.dtype}"
        )


        # ==================================================
        # GUARDAR IMAGEN ORIGINAL
        # ==================================================

        guardar_debug(
            "01_imagen_original.jpg",
            img
        )


        # ==================================================
        # INTENTAR DEEPFACE
        # ==================================================

        try:

            logger.info(
                "🔎 Intentando detectar rostro con DeepFace..."
            )


            DF = cargar_deepface()


            if DF is not None:

                faces = DF.extract_faces(
                    img_path=img,
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=True
                )


                logger.info(
                    f"👤 DeepFace detectó {len(faces)} rostro(s)"
                )


                if len(faces) > 0:

                    rostro = faces[0]["face"]


                    logger.info(
                        f"📷 Rostro DeepFace: {rostro.shape}"
                    )


                    # ==========================================
                    # DEEPFACE PUEDE DEVOLVER VALORES 0-1
                    # ==========================================

                    if rostro.max() <= 1.0:

                        rostro = (
                            rostro * 255
                        ).astype(np.uint8)


                    # ==========================================
                    # CONVERTIR RGB → BGR
                    # ==========================================

                    if len(rostro.shape) == 3:

                        rostro = cv2.cvtColor(
                            rostro,
                            cv2.COLOR_RGB2BGR
                        )


                    # ==========================================
                    # GUARDAR ROSTRO
                    # ==========================================

                    guardar_debug(
                        "02_rostro_deepface.jpg",
                        rostro
                    )


                    logger.info(
                        f"✅ Rostro obtenido con DeepFace: "
                        f"{rostro.shape}"
                    )


                    logger.info(
                        "➡️ Enviando rostro a predict.py"
                    )


                    return rostro


        except Exception as e:

            logger.warning(
                f"⚠️ DeepFace no pudo detectar el rostro: {e}"
            )

            logger.info(
                "🔄 Cambiando a OpenCV Haar..."
            )


        # ==================================================
        # OPENCV COMO RESPALDO
        # ==================================================

        logger.info(
            "🔎 Detectando rostro con OpenCV Haar..."
        )


        # ==================================================
        # CONVERTIR A GRIS
        # ==================================================

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        # ==================================================
        # DETECTAR ROSTROS
        # ==================================================

        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        # ==================================================
        # VERIFICAR
        # ==================================================

        if len(caras) == 0:

            logger.warning(
                "❌ OpenCV tampoco detectó ningún rostro"
            )

            return None


        logger.info(
            f"👤 OpenCV detectó {len(caras)} rostro(s)"
        )


        # ==================================================
        # SELECCIONAR EL ROSTRO MÁS GRANDE
        # ==================================================

        x, y, w, h = max(
            caras,
            key=lambda r: r[2] * r[3]
        )


        logger.info(
            f"📐 Rostro: x={x}, y={y}, w={w}, h={h}"
        )


        # ==================================================
        # MARGEN
        # ==================================================

        margen = int(
            0.15 * max(w, h)
        )


        x1 = max(
            0,
            x - margen
        )

        y1 = max(
            0,
            y - margen
        )

        x2 = min(
            img.shape[1],
            x + w + margen
        )

        y2 = min(
            img.shape[0],
            y + h + margen
        )


        # ==================================================
        # RECORTAR ROSTRO
        # ==================================================

        rostro = img[
            y1:y2,
            x1:x2
        ]


        if rostro is None or rostro.size == 0:

            logger.warning(
                "❌ El recorte está vacío"
            )

            return None


        logger.info(
            f"📷 Rostro OpenCV: {rostro.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO
        # ==================================================

        guardar_debug(
            "02_rostro_opencv.jpg",
            rostro
        )


        logger.info(
            "✅ Rostro detectado con OpenCV"
        )


        logger.info(
            "➡️ Enviando rostro a predict.py"
        )


        return rostro


    except Exception as e:

        logger.exception(
            f"❌ Error en detectar_rostro(): {e}"
        )

        return None