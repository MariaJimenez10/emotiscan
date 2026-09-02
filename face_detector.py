import os
import cv2
import logging
import numpy as np


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================
# CARPETA DE DEBUG
# ==========================================================

DEBUG_DIR = os.path.join(
    BASE_DIR,
    "debug_rostros"
)

os.makedirs(
    DEBUG_DIR,
    exist_ok=True
)


# ==========================================================
# HAAR CASCADE
# ==========================================================

HAAR_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    HAAR_PATH
)


if face_cascade.empty():

    logger.error(
        "❌ No se pudo cargar Haar Cascade: %s",
        HAAR_PATH
    )

else:

    logger.info(
        "✅ Haar Cascade cargado correctamente"
    )


# ==========================================================
# GUARDAR IMAGEN DE DEPURACIÓN
# ==========================================================

def guardar_debug(nombre, imagen):
    """
    Guarda una imagen para verificar qué rostro
    fue detectado y recortado.

    Esto NO participa en la predicción.
    Solo sirve para depuración.
    """

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

        resultado = cv2.imwrite(
            ruta,
            imagen
        )

        if resultado:

            logger.info(
                "💾 Imagen de debug guardada: %s",
                ruta
            )

        else:

            logger.warning(
                "⚠️ OpenCV no pudo guardar: %s",
                ruta
            )

    except Exception as e:

        logger.warning(
            "⚠️ Error guardando imagen de debug: %s",
            e
        )


# ==========================================================
# PREPROCESAR IMAGEN
# ==========================================================

def preparar_imagen(img):
    """
    Verifica que la imagen sea válida y esté
    en formato BGR de OpenCV.
    """

    if img is None:

        logger.error(
            "❌ La imagen recibida es None"
        )

        return None

    if not isinstance(img, np.ndarray):

        logger.error(
            "❌ La imagen no es numpy.ndarray"
        )

        return None

    if img.size == 0:

        logger.error(
            "❌ La imagen está vacía"
        )

        return None

    # ------------------------------------------------------
    # Si viene en escala de grises
    # ------------------------------------------------------

    if len(img.shape) == 2:

        logger.info(
            "🔄 Imagen en escala de grises. Convirtiendo a BGR..."
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    # ------------------------------------------------------
    # Si tiene canal alfa
    # ------------------------------------------------------

    elif len(img.shape) == 3 and img.shape[2] == 4:

        logger.info(
            "🔄 Imagen BGRA. Eliminando canal alfa..."
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGRA2BGR
        )

    # ------------------------------------------------------
    # Validar canales
    # ------------------------------------------------------

    if len(img.shape) != 3 or img.shape[2] != 3:

        logger.error(
            "❌ Formato de imagen no compatible: %s",
            img.shape
        )

        return None

    return img


# ==========================================================
# DETECTAR ROSTRO
# ==========================================================

def detectar_con_haar(img):
    """
    Detecta rostros utilizando Haar Cascade.

    Si encuentra varios rostros, selecciona el más grande.

    Devuelve únicamente el rostro recortado.
    """

    try:

        logger.info(
            "🔎 Iniciando detección Haar Cascade..."
        )

        # ==================================================
        # VALIDAR IMAGEN
        # ==================================================

        img = preparar_imagen(img)

        if img is None:
            return None

        logger.info(
            "📷 Imagen para detección: %s",
            img.shape
        )

        # ==================================================
        # CONVERTIR A ESCALA DE GRISES
        # ==================================================

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # ==================================================
        # MEJORAR CONTRASTE
        # ==================================================

        gray = cv2.equalizeHist(
            gray
        )

        # ==================================================
        # DETECTAR ROSTROS
        # ==================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if faces is None or len(faces) == 0:

            logger.warning(
                "⚠️ Haar Cascade no encontró ningún rostro"
            )

            return None

        logger.info(
            "👤 Haar Cascade encontró %d rostro(s)",
            len(faces)
        )

        # ==================================================
        # SELECCIONAR EL ROSTRO MÁS GRANDE
        # ==================================================

        x, y, w, h = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )

        logger.info(
            "📐 Rostro seleccionado:"
            " x=%d y=%d w=%d h=%d",
            x,
            y,
            w,
            h
        )

        # ==================================================
        # AGREGAR MARGEN
        # ==================================================

        margen_x = int(w * 0.15)
        margen_y = int(h * 0.20)

        x1 = max(
            0,
            x - margen_x
        )

        y1 = max(
            0,
            y - margen_y
        )

        x2 = min(
            img.shape[1],
            x + w + margen_x
        )

        y2 = min(
            img.shape[0],
            y + h + margen_y
        )

        logger.info(
            "✂️ Coordenadas finales:"
            " x1=%d y1=%d x2=%d y2=%d",
            x1,
            y1,
            x2,
            y2
        )

        # ==================================================
        # RECORTAR ROSTRO
        # ==================================================

        rostro = img[
            y1:y2,
            x1:x2
        ]

        # ==================================================
        # VALIDAR RECORTE
        # ==================================================

        if rostro is None:

            logger.error(
                "❌ El recorte devolvió None"
            )

            return None

        if rostro.size == 0:

            logger.error(
                "❌ El recorte del rostro está vacío"
            )

            return None

        logger.info(
            "✅ Rostro recortado correctamente: %s",
            rostro.shape
        )

        # ==================================================
        # GUARDAR ROSTRO
        # ==================================================

        guardar_debug(
            "rostro_detectado.jpg",
            rostro
        )

        return rostro

    except Exception as e:

        logger.exception(
            "❌ Error detectando rostro con Haar: %s",
            e
        )

        return None


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def detectar_rostro(img):
    """
    Detecta y recorta el rostro.

    Flujo:

        Imagen original
              ↓
        Validación
              ↓
        Haar Cascade
              ↓
        Selección del rostro más grande
              ↓
        Recorte
              ↓
        Guardar debug
              ↓
        Retornar rostro

    IMPORTANTE:

    Este archivo NO predice emociones.

    El rostro retornado debe enviarse posteriormente
    al modelo de emociones.
    """

    logger.info(
        "=========================================="
    )

    logger.info(
        "👤 INICIANDO DETECCIÓN DE ROSTRO"
    )

    logger.info(
        "=========================================="
    )

    # ======================================================
    # VALIDAR IMAGEN
    # ======================================================

    img = preparar_imagen(img)

    if img is None:

        logger.error(
            "❌ Imagen inválida"
        )

        return None

    # ======================================================
    # INFORMACIÓN DE LA IMAGEN
    # ======================================================

    logger.info(
        "📷 Imagen recibida correctamente"
    )

    logger.info(
        "📐 Dimensiones: %s",
        img.shape
    )

    logger.info(
        "💾 Tipo: %s",
        img.dtype
    )

    # ======================================================
    # GUARDAR ORIGINAL
    # ======================================================

    guardar_debug(
        "imagen_original.jpg",
        img
    )

    # ======================================================
    # DETECTAR ROSTRO
    # ======================================================

    rostro = detectar_con_haar(
        img
    )

    # ======================================================
    # RESULTADO
    # ======================================================

    if rostro is not None:

        logger.info(
            "=========================================="
        )

        logger.info(
            "✅ ROSTRO DETECTADO CORRECTAMENTE"
        )

        logger.info(
            "📐 Tamaño: %s",
            rostro.shape
        )

        logger.info(
            "=========================================="
        )

        return rostro

    # ======================================================
    # NO ENCONTRADO
    # ======================================================

    logger.error(
        "=========================================="
    )

    logger.error(
        "❌ NO SE ENCONTRÓ NINGÚN ROSTRO"
    )

    logger.error(
        "=========================================="
    )

    return None


# ==========================================================
# PRUEBA LOCAL
# ==========================================================

if __name__ == "__main__":

    logger.info(
        "=========================================="
    )

    logger.info(
        "   PRUEBA DEL DETECTOR DE ROSTROS"
    )

    logger.info(
        "=========================================="
    )

    # ------------------------------------------------------
    # Imagen de prueba
    # ------------------------------------------------------

    ruta_imagen = os.path.join(
        BASE_DIR,
        "test.jpg"
    )

    logger.info(
        "📂 Imagen de prueba: %s",
        ruta_imagen
    )

    imagen = cv2.imread(
        ruta_imagen
    )

    # ------------------------------------------------------
    # Verificar imagen
    # ------------------------------------------------------

    if imagen is None:

        logger.error(
            "❌ No se encontró la imagen de prueba"
        )

        logger.error(
            "📂 Ruta: %s",
            ruta_imagen
        )

    else:

        logger.info(
            "✅ Imagen de prueba cargada"
        )

        logger.info(
            "📐 Tamaño: %s",
            imagen.shape
        )

        # --------------------------------------------------
        # Detectar rostro
        # --------------------------------------------------

        rostro = detectar_rostro(
            imagen
        )

        # --------------------------------------------------
        # Resultado
        # --------------------------------------------------

        if rostro is not None:

            logger.info(
                "=========================================="
            )

            logger.info(
                "✅ PRUEBA EXITOSA"
            )

            logger.info(
                "📐 Tamaño del rostro: %s",
                rostro.shape
            )

            logger.info(
                "📂 Rostro guardado en:"
            )

            logger.info(
                "%s",
                DEBUG_DIR
            )

            logger.info(
                "=========================================="
            )

        else:

            logger.error(
                "=========================================="
            )

            logger.error(
                "❌ PRUEBA FALLIDA"
            )

            logger.error(
                "=========================================="
            )