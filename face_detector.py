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
<<<<<<< HEAD
# CARGAR HAAR CASCADE
# ==========================================================

HAAR_PATH = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"
=======
# HAAR CASCADE
# ==========================================================

HAAR_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

face_cascade = cv2.CascadeClassifier(
    HAAR_PATH
)

<<<<<<< HEAD
if face_cascade.empty():

    logger.error(
        "❌ No se pudo cargar Haar Cascade"
    )

    raise RuntimeError(
        "No se pudo cargar haarcascade_frontalface_default.xml"
=======

if face_cascade.empty():

    logger.error(
        "❌ No se pudo cargar Haar Cascade: %s",
        HAAR_PATH
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    )

else:

    logger.info(
        "✅ Haar Cascade cargado correctamente"
    )


# ==========================================================
<<<<<<< HEAD
# GUARDAR IMAGEN DE DEBUG
# ==========================================================

def guardar_debug(nombre, imagen):
=======
# GUARDAR IMAGEN DE DEPURACIÓN
# ==========================================================

def guardar_debug(nombre, imagen):
    """
    Guarda una imagen para verificar qué rostro
    fue detectado y recortado.

    Esto NO participa en la predicción.
    Solo sirve para depuración.
    """
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    try:

        if imagen is None:
<<<<<<< HEAD

            logger.warning(
                "⚠️ No se puede guardar debug: imagen None"
            )

=======
            return

        if not isinstance(imagen, np.ndarray):
            return

        if imagen.size == 0:
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            return

        ruta = os.path.join(
            DEBUG_DIR,
            nombre
        )

<<<<<<< HEAD
        cv2.imwrite(
=======
        resultado = cv2.imwrite(
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            ruta,
            imagen
        )

<<<<<<< HEAD
        logger.info(
            "💾 Debug guardado: %s",
            ruta
        )
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    except Exception as e:

        logger.warning(
<<<<<<< HEAD
            "⚠️ No se pudo guardar debug: %s",
=======
            "⚠️ Error guardando imagen de debug: %s",
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            e
        )


# ==========================================================
<<<<<<< HEAD
# PREPARAR IMAGEN
# ==========================================================

def preparar_imagen(img):

    try:

        if img is None:

            logger.error(
                "❌ La imagen recibida es None"
            )

            return None

        if not isinstance(img, np.ndarray):

            logger.error(
                "❌ La imagen no es un numpy.ndarray"
            )

            return None

        if img.size == 0:

            logger.error(
                "❌ La imagen está vacía"
            )

            return None

        # Imagen en escala de grises
        if len(img.shape) == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

        # Imagen RGBA
        elif len(img.shape) == 3 and img.shape[2] == 4:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2BGR
            )

        # Imagen BGR normal
        elif len(img.shape) == 3 and img.shape[2] == 3:

            pass

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
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        )

        return None

<<<<<<< HEAD

# ==========================================================
# DETECCIÓN HAAR
# ==========================================================

def detectar_con_haar(img):
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    try:

        logger.info(
            "🔎 Iniciando detección Haar Cascade..."
        )

<<<<<<< HEAD
        # --------------------------------------------------
        # PREPARAR IMAGEN
        # --------------------------------------------------
=======
        # ==================================================
        # VALIDAR IMAGEN
        # ==================================================
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        img = preparar_imagen(img)

        if img is None:
<<<<<<< HEAD

            return None

        # --------------------------------------------------
        # ESCALA DE GRISES
        # --------------------------------------------------
=======
            return None

        logger.info(
            "📷 Imagen para detección: %s",
            img.shape
        )

        # ==================================================
        # CONVERTIR A ESCALA DE GRISES
        # ==================================================
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

<<<<<<< HEAD
        # --------------------------------------------------
        # SUAVIZADO
        # --------------------------------------------------

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # --------------------------------------------------
        # DETECCIÓN
        # --------------------------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(60, 60),
            maxSize=(500, 500)
        )

        if len(faces) == 0:

            logger.error(
                "❌ No se encontró ningún rostro"
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            )

            return None

        logger.info(
<<<<<<< HEAD
            "👤 Detecciones encontradas: %d",
            len(faces)
        )

        # --------------------------------------------------
        # MOSTRAR DETECCIONES
        # --------------------------------------------------

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
        # ELIMINAR DETECCIONES DUPLICADAS
        # ==================================================

        rectangulos = []

        for (x, y, w, h) in faces:

            rectangulos.append(
                [int(x), int(y), int(w), int(h)]
            )

        # Haar puede detectar varias veces el mismo rostro.
        # groupRectangles permite agrupar esas detecciones.
        rectangulos_agrupados, pesos = cv2.groupRectangles(
            rectangulos,
            groupThreshold=1,
            eps=0.4
        )

        # Si el agrupamiento elimina todo, usamos las
        # detecciones originales.
        if len(rectangulos_agrupados) > 0:

            faces_finales = rectangulos_agrupados

            logger.info(
                "🔗 Detecciones agrupadas: %d",
                len(faces_finales)
            )

        else:

            faces_finales = np.array(
                faces
            )

            logger.info(
                "ℹ️ Se utilizarán las detecciones originales"
            )

        # ==================================================
        # SELECCIONAR EL ROSTRO PRINCIPAL
        # ==================================================

        alto_img, ancho_img = img.shape[:2]

        centro_img_x = ancho_img / 2
        centro_img_y = alto_img / 2

        candidatos = []

        for rect in faces_finales:

            x, y, w, h = map(
                int,
                rect
            )

            centro_x = x + w / 2
            centro_y = y + h / 2

            distancia = np.sqrt(
                (centro_x - centro_img_x) ** 2 +
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

        # Primero priorizamos rostros razonablemente grandes
        candidatos_validos = [
            c for c in candidatos
            if c["w"] >= 80 and c["h"] >= 80
        ]

        if len(candidatos_validos) == 0:

            logger.warning(
                "⚠️ No hay rostros con tamaño suficiente"
            )

            return None

        # Seleccionar el candidato más cercano al centro,
        # pero dando importancia también al tamaño.
        #
        # Esto evita que una detección gigante y desplazada
        # gane simplemente por tener mayor área.

        mejor = min(
            candidatos_validos,
            key=lambda c: (
                c["distancia"] / max(c["w"], c["h"])
            )
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

        # --------------------------------------------------
        # MARGEN DEL 5%
        # --------------------------------------------------

        margen_x = int(
            w * 0.05
        )

        margen_y = int(
            h * 0.05
        )

        # --------------------------------------------------
        # COORDENADAS FINALES
        # --------------------------------------------------
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

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
<<<<<<< HEAD
            "✂️ Coordenadas finales: "
            "x1=%d y1=%d x2=%d y2=%d",
=======
            "✂️ Coordenadas finales:"
            " x1=%d y1=%d x2=%d y2=%d",
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            x1,
            y1,
            x2,
            y2
        )

<<<<<<< HEAD
        # --------------------------------------------------
        # RECORTAR ROSTRO
        # --------------------------------------------------
=======
        # ==================================================
        # RECORTAR ROSTRO
        # ==================================================
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        rostro = img[
            y1:y2,
            x1:x2
<<<<<<< HEAD
        ].copy()

        if rostro is None or rostro.size == 0:

            logger.error(
                "❌ El recorte del rostro está vacío"
=======
        ]

        # ==================================================
        # VALIDAR RECORTE
        # ==================================================

        if rostro is None:

            logger.error(
                "❌ El recorte devolvió None"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            )

            return None

<<<<<<< HEAD
        alto = rostro.shape[0]
        ancho = rostro.shape[1]

        if ancho < 80 or alto < 80:

            logger.warning(
                "⚠️ Recorte demasiado pequeño: %dx%d",
                ancho,
                alto
=======
        if rostro.size == 0:

            logger.error(
                "❌ El recorte del rostro está vacío"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            )

            return None

        logger.info(
            "✅ Rostro recortado correctamente: %s",
            rostro.shape
        )

<<<<<<< HEAD
        # --------------------------------------------------
        # GUARDAR ROSTRO
        # --------------------------------------------------
=======
        # ==================================================
        # GUARDAR ROSTRO
        # ==================================================
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        guardar_debug(
            "rostro_detectado.jpg",
            rostro
        )

<<<<<<< HEAD
        # --------------------------------------------------
        # DIBUJAR RECTÁNGULO
        # --------------------------------------------------

        img_con_rect = img.copy()

        for rect in faces_finales:

            fx, fy, fw, fh = map(
                int,
                rect
            )

            # Rectángulo de todas las detecciones
            cv2.rectangle(
                img_con_rect,
                (fx, fy),
                (fx + fw, fy + fh),
                (255, 0, 0),
                2
            )

        # Rectángulo del rostro seleccionado
        cv2.rectangle(
            img_con_rect,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

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

=======
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        return rostro

    except Exception as e:

        logger.exception(
<<<<<<< HEAD
            "❌ Error detectando rostro: %s",
=======
            "❌ Error detectando rostro con Haar: %s",
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
            e
        )

        return None


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def detectar_rostro(img):
<<<<<<< HEAD

    try:

        logger.info(
            "🚀 Iniciando proceso de detección facial..."
        )

        rostro = detectar_con_haar(
            img
        )

        if rostro is None:

            logger.error(
                "❌ No fue posible detectar un rostro"
            )

            return None

        logger.info(
            "✅ Proceso de detección terminado correctamente"
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        )

        return rostro

<<<<<<< HEAD
    except Exception as e:

        logger.exception(
            "❌ Error general en detectar_rostro: %s",
            e
        )

        return None
=======
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
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
