import os
import cv2
import logging
import numpy as np

from deepface import DeepFace


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ==========================================================
# CARPETA PARA DIAGNÓSTICO
# ==========================================================

DEBUG_DIR = os.path.join(
    os.path.dirname(__file__),
    "debug_rostros"
)

os.makedirs(DEBUG_DIR, exist_ok=True)


# ==========================================================
# DETECTOR OPENCV DE RESPALDO
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================================
# GUARDAR IMAGEN DE DEBUG
# ==========================================================

def guardar_debug(nombre, imagen):

    try:

        if imagen is None:
            logger.warning(
                f"⚠️ No se puede guardar {nombre}: imagen None"
            )
            return

        ruta = os.path.join(
            DEBUG_DIR,
            nombre
        )

        resultado = cv2.imwrite(
            ruta,
            imagen
        )

        if resultado:

            logger.info(
                f"📸 Imagen guardada: {ruta}"
            )

        else:

            logger.warning(
                f"⚠️ OpenCV no pudo guardar: {ruta}"
            )

    except Exception as e:

        logger.warning(
            f"⚠️ No se pudo guardar {nombre}: {e}"
        )


# ==========================================================
# DETECTAR ROSTRO
# ==========================================================

def detectar_rostro(img):

    print("🔥 detectar_rostro() FUE LLAMADA")

    try:

        # ==================================================
        # VALIDAR IMAGEN
        # ==================================================

        if img is None:

            logger.error(
                "❌ La imagen recibida es None."
            )

            return None


        if not isinstance(img, np.ndarray):

            logger.error(
                f"❌ La imagen no es numpy.ndarray: "
                f"{type(img)}"
            )

            return None


        if img.size == 0:

            logger.error(
                "❌ La imagen está vacía."
            )

            return None


        logger.info(
            "🔍 INICIANDO DETECCIÓN DE ROSTRO"
        )

        logger.info(
            f"Imagen original: {img.shape}"
        )

        logger.info(
            f"Tipo imagen: {img.dtype}"
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


            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="opencv",
                enforce_detection=True,
                align=True
            )


            logger.info(
                f"DeepFace detectó {len(faces)} rostro(s)"
            )


            if len(faces) > 0:

                # ==================================================
                # TOMAR PRIMER ROSTRO
                # ==================================================

                rostro = faces[0]["face"]


                logger.info(
                    f"Rostro DeepFace original: "
                    f"{rostro.shape}"
                )


                # ==================================================
                # DEEPFACE PUEDE DEVOLVER VALORES 0-1
                # ==================================================

                if rostro.max() <= 1:

                    rostro = (
                        rostro * 255
                    ).astype(np.uint8)


                # ==================================================
                # RGB → BGR
                # ==================================================

                if len(rostro.shape) == 3:

                    rostro = cv2.cvtColor(
                        rostro,
                        cv2.COLOR_RGB2BGR
                    )


                # ==================================================
                # GUARDAR ROSTRO DETECTADO
                # ==================================================

                guardar_debug(
                    "02_rostro_deepface_original.jpg",
                    rostro
                )


                # ==================================================
                # REDIMENSIONAR
                # ==================================================

                rostro = cv2.resize(
                    rostro,
                    (224, 224),
                    interpolation=cv2.INTER_AREA
                )


                logger.info(
                    "✅ Rostro detectado con DeepFace"
                )

                logger.info(
                    f"Rostro final: {rostro.shape}"
                )


                # ==================================================
                # GUARDAR ROSTRO ANTES DEL MODELO
                # ==================================================

                guardar_debug(
                    "03_rostro_antes_modelo.jpg",
                    rostro
                )


                logger.info(
                    "======================================"
                )


                return rostro


        except Exception as e:

            logger.warning(
                f"⚠️ DeepFace falló: {e}"
            )

            logger.info(
                "🔄 Utilizando detector OpenCV..."
            )


        # ==================================================
        # OPENCV - RESPALDO
        # ==================================================

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

            logger.warning(
                "❌ No se detectó ningún rostro."
            )

            return None


        logger.info(
            f"👤 Rostros detectados con OpenCV: "
            f"{len(caras)}"
        )


        # ==================================================
        # SELECCIONAR ROSTRO MÁS GRANDE
        # ==================================================

        x, y, w, h = max(
            caras,
            key=lambda r: r[2] * r[3]
        )


        logger.info(
            f"Rostro detectado: "
            f"x={x}, y={y}, w={w}, h={h}"
        )


        # ==================================================
        # AGREGAR MARGEN
        # ==================================================

        margen = int(
            0.15 * max(w, h)
        )


        x = max(
            0,
            x - margen
        )

        y = max(
            0,
            y - margen
        )

        w = min(
            img.shape[1] - x,
            w + margen * 2
        )

        h = min(
            img.shape[0] - y,
            h + margen * 2
        )


        # ==================================================
        # RECORTAR ROSTRO
        # ==================================================

        rostro = img[
            y:y + h,
            x:x + w
        ]


        if rostro.size == 0:

            logger.warning(
                "❌ El recorte del rostro está vacío."
            )

            return None


        logger.info(
            f"Rostro recortado: {rostro.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO OPENCV
        # ==================================================

        guardar_debug(
            "02_rostro_opencv_original.jpg",
            rostro
        )


        # ==================================================
        # REDIMENSIONAR
        # ==================================================

        rostro = cv2.resize(
            rostro,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )


        logger.info(
            f"Rostro final: {rostro.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO ANTES DEL MODELO
        # ==================================================

        guardar_debug(
            "03_rostro_antes_modelo.jpg",
            rostro
        )


        logger.info(
            "✅ Rostro detectado con OpenCV"
        )

        logger.info(
            "======================================"
        )


        return rostro


    except Exception as e:

        logger.exception(
            f"❌ Error detectando rostro: {e}"
        )

        return None