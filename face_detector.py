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
# CARGAR HAAR CASCADE
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

    raise RuntimeError(
        "No se pudo cargar haarcascade_frontalface_default.xml"
    )

else:

    logger.info(
        "✅ Haar Cascade cargado correctamente"
    )


# ==========================================================
# GUARDAR IMAGEN DE DEBUG
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

            logger.warning(
                "⚠️ No se puede guardar debug: imagen None"
            )

            return

        if not isinstance(imagen, np.ndarray):

            logger.warning(
                "⚠️ No se puede guardar debug: formato inválido"
            )

            return

        if imagen.size == 0:

            logger.warning(
                "⚠️ No se puede guardar debug: imagen vacía"
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
# PREPARAR IMAGEN
# ==========================================================

def preparar_imagen(img):
    """
    Verifica que la imagen sea válida y la convierte
    al formato BGR utilizado por OpenCV.
    """

    try:

        # --------------------------------------------------
        # Validar None
        # --------------------------------------------------

        if img is None:

            logger.error(
                "❌ La imagen recibida es None"
            )

            return None

        # --------------------------------------------------
        # Validar numpy
        # --------------------------------------------------

        if not isinstance(img, np.ndarray):

            logger.error(
                "❌ La imagen no es un numpy.ndarray"
            )

            return None

        # --------------------------------------------------
        # Validar imagen vacía
        # --------------------------------------------------

        if img.size == 0:

            logger.error(
                "❌ La imagen está vacía"
            )

            return None

        # --------------------------------------------------
        # Imagen en escala de grises
        # --------------------------------------------------

        if len(img.shape) == 2:

            logger.info(
                "🔄 Imagen en escala de grises. "
                "Convirtiendo a BGR..."
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

        # --------------------------------------------------
        # Imagen RGBA / BGRA
        # --------------------------------------------------

        elif len(img.shape) == 3 and img.shape[2] == 4:

            logger.info(
                "🔄 Imagen BGRA. Eliminando canal alfa..."
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2BGR
            )

        # --------------------------------------------------
        # Imagen BGR normal
        # --------------------------------------------------

        elif len(img.shape) == 3 and img.shape[2] == 3:

            pass

        # --------------------------------------------------
        # Formato no compatible
        # --------------------------------------------------

        else:

            logger.error(
                "❌ Formato de imagen no soportado: %s",
                img.shape
            )

            return None

        logger.info(
            "📷 Imagen preparada: %s",
            img.shape
        )

        return img

    except Exception as e:

        logger.exception(
            "❌ Error preparando imagen: %s",
            e
        )

        return None


# ==========================================================
# DETECTAR ROSTRO CON HAAR
# ==========================================================

def detectar_con_haar(img):
    """
    Detecta rostros utilizando Haar Cascade.

    Si encuentra varios rostros:
    1. Agrupa detecciones duplicadas.
    2. Descarta rostros demasiado pequeños.
    3. Prioriza el rostro más cercano al centro.
    4. Recorta el rostro con un margen.

    Retorna:
        rostro recortado como numpy.ndarray

    Si no encuentra rostro:
        None
    """

    try:

        logger.info(
            "🔎 Iniciando detección Haar Cascade..."
        )

        # ==================================================
        # PREPARAR IMAGEN
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
        # SUAVIZADO
        # ==================================================

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # ==================================================
        # DETECTAR ROSTROS
        # ==================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(60, 60),
            maxSize=(500, 500)
        )

        # ==================================================
        # VALIDAR DETECCIONES
        # ==================================================

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
        # MOSTRAR DETECCIONES
        # ==================================================

        for i, (x, y, w, h) in enumerate(faces):

            logger.info(
                "👤 Detección %d: "
                "x=%d y=%d w=%d h=%d área=%d",
                i + 1,
                x,
                y,
                w,
                h,
                w * h
            )

        # ==================================================
        # CONVERTIR A LISTA DE RECTÁNGULOS
        # ==================================================

        rectangulos = []

        for (x, y, w, h) in faces:

            rectangulos.append(
                [
                    int(x),
                    int(y),
                    int(w),
                    int(h)
                ]
            )

        # ==================================================
        # AGRUPAR DETECCIONES DUPLICADAS
        # ==================================================

        rectangulos_agrupados, pesos = cv2.groupRectangles(
            rectangulos,
            groupThreshold=1,
            eps=0.4
        )

        # ==================================================
        # SELECCIONAR DETECCIONES FINALES
        # ==================================================

        if (
            rectangulos_agrupados is not None
            and len(rectangulos_agrupados) > 0
        ):

            faces_finales = rectangulos_agrupados

            logger.info(
                "🔗 Detecciones agrupadas: %d",
                len(faces_finales)
            )

        else:

            faces_finales = np.array(
                rectangulos
            )

            logger.info(
                "ℹ️ Se utilizarán las detecciones originales"
            )

        # ==================================================
        # INFORMACIÓN DE LA IMAGEN
        # ==================================================

        alto_img, ancho_img = img.shape[:2]

        centro_img_x = ancho_img / 2
        centro_img_y = alto_img / 2

        candidatos = []

        # ==================================================
        # ANALIZAR CANDIDATOS
        # ==================================================

        for rect in faces_finales:

            x, y, w, h = map(
                int,
                rect
            )

            centro_x = x + w / 2
            centro_y = y + h / 2

            distancia = np.sqrt(
                (centro_x - centro_img_x) ** 2
                +
                (centro_y - centro_img_y) ** 2
            )

            area = w * h

            candidatos.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "distancia": distancia
                }
            )

        # ==================================================
        # FILTRAR ROSTROS PEQUEÑOS
        # ==================================================

        candidatos_validos = [
            c
            for c in candidatos
            if c["w"] >= 80
            and c["h"] >= 80
        ]

        if len(candidatos_validos) == 0:

            logger.warning(
                "⚠️ No hay rostros con tamaño suficiente"
            )

            return None

        # ==================================================
        # SELECCIONAR ROSTRO PRINCIPAL
        # ==================================================

        # Seleccionamos el rostro más grande.
        # Normalmente corresponde a la persona que está
        # más cerca de la cámara.

        mejor = max(
        candidatos_validos,
        key=lambda c: c["area"]
        )
        x = mejor["x"]
        y = mejor["y"]
        w = mejor["w"]
        h = mejor["h"]

        logger.info(
            "🎯 Rostro seleccionado: "
            "x=%d y=%d w=%d h=%d área=%d",
            x,
            y,
            w,
            h,
            w * h
        )

        # ==================================================
        # AGREGAR MARGEN
        # ==================================================
        margen_x = int(w * 0.05)
        margen_y = int(h * 0.08)
        # ==================================================
        # COORDENADAS FINALES
        # ==================================================

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
            "✂️ Coordenadas finales: "
            "x1=%d y1=%d x2=%d y2=%d",
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
        ].copy()

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

        alto = rostro.shape[0]
        ancho = rostro.shape[1]

        if ancho < 80 or alto < 80:

            logger.warning(
                "⚠️ Recorte demasiado pequeño: %dx%d",
                ancho,
                alto
            )

            return None

        logger.info(
            "✅ Rostro recortado correctamente: %s",
            rostro.shape
        )

        # ==================================================
        # GUARDAR ROSTRO DETECTADO
        # ==================================================

        guardar_debug(
            "rostro_detectado.jpg",
            rostro
        )

        # ==================================================
        # DIBUJAR RECTÁNGULOS
        # ==================================================

        img_con_rect = img.copy()

        # --------------------------------------------------
        # Rectángulos de todas las detecciones
        # --------------------------------------------------

        for rect in faces_finales:

            fx, fy, fw, fh = map(
                int,
                rect
            )

            cv2.rectangle(
                img_con_rect,
                (fx, fy),
                (fx + fw, fy + fh),
                (255, 0, 0),
                2
            )

        # --------------------------------------------------
        # Rectángulo del rostro seleccionado
        # --------------------------------------------------

        cv2.rectangle(
            img_con_rect,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # ==================================================
        # GUARDAR DEBUG CON RECTÁNGULOS
        # ==================================================

        guardar_debug(
            "rostro_con_rectangulo.jpg",
            img_con_rect
        )

        logger.info(
            "✅ ROSTRO DETECTADO CORRECTAMENTE"
        )

        logger.info(
            "📐 Tamaño final: %dx%d",
            rostro.shape[1],
            rostro.shape[0]
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
    Función principal para detectar y recortar un rostro.

    Flujo:

        Imagen original
              ↓
        Validación
              ↓
        Haar Cascade
              ↓
        Selección del rostro principal
              ↓
        Recorte
              ↓
        Guardar debug
              ↓
        Retornar rostro

    IMPORTANTE:

    Este archivo NO predice emociones.

    El rostro retornado debe enviarse posteriormente
    al modelo de reconocimiento de emociones.
    """

    try:

        logger.info(
            "=========================================="
        )

        logger.info(
            "👤 INICIANDO DETECCIÓN DE ROSTRO"
        )

        logger.info(
            "=========================================="
        )

        # ==================================================
        # PREPARAR IMAGEN
        # ==================================================

        img = preparar_imagen(
            img
        )

        if img is None:

            logger.error(
                "❌ Imagen inválida"
            )

            return None

        # ==================================================
        # INFORMACIÓN DE LA IMAGEN
        # ==================================================

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

        # ==================================================
        # GUARDAR ORIGINAL
        # ==================================================

        guardar_debug(
            "imagen_original.jpg",
            img
        )

        # ==================================================
        # DETECTAR ROSTRO
        # ==================================================

        rostro = detectar_con_haar(
            img
        )

        # ==================================================
        # ROSTRO ENCONTRADO
        # ==================================================

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

        # ==================================================
        # NO ENCONTRADO
        # ==================================================

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

    except Exception as e:

        logger.exception(
            "❌ Error general en detectar_rostro: %s",
            e
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

    # ======================================================
    # IMAGEN DE PRUEBA
    # ======================================================

    ruta_imagen = os.path.join(
        BASE_DIR,
        "test.jpg"
    )

    logger.info(
        "📂 Imagen de prueba: %s",
        ruta_imagen
    )

    # ======================================================
    # CARGAR IMAGEN
    # ======================================================

    imagen = cv2.imread(
        ruta_imagen
    )

    # ======================================================
    # VERIFICAR IMAGEN
    # ======================================================

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

        # ==================================================
        # DETECTAR ROSTRO
        # ==================================================

        rostro = detectar_rostro(
            imagen
        )

        # ==================================================
        # RESULTADO
        # ==================================================

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
                "📂 Rostros de debug guardados en:"
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