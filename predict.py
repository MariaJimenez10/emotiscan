import cv2
import os
import logging


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_emociones.xml"
)

DEBUG_DIR = os.path.join(
    BASE_DIR,
    "debug_rostros"
)

os.makedirs(
    DEBUG_DIR,
    exist_ok=True
)


# ==========================================================
# EMOCIONES
#
# MISMO ORDEN UTILIZADO EN entrenamiento.py
# ==========================================================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# TAMAÑO DEL ROSTRO
#
# Tus imágenes de entrenamiento son 48x48
# ==========================================================

IMG_SIZE = (48, 48)


# ==========================================================
# CARGAR MODELO LBPH
# ==========================================================

logger.info("==========================================")
logger.info("       CARGANDO MODELO DE EMOCIONES")
logger.info("==========================================")

logger.info(
    f"📁 Ubicación del modelo: {MODEL_PATH}"
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"❌ No se encontró el modelo:\n{MODEL_PATH}"
    )


# Crear reconocedor LBPH
emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()


# Cargar modelo XML
emotion_recognizer.read(MODEL_PATH)


logger.info("✅ Modelo LBPH cargado correctamente")


# ==========================================================
# CARGAR DETECTOR DE ROSTROS
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


if face_cascade.empty():

    raise RuntimeError(
        "❌ No se pudo cargar el detector Haar Cascade"
    )


logger.info("✅ Detector de rostros cargado")
logger.info("==========================================")


# ==========================================================
# FUNCIÓN PARA PREDECIR EMOCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info("")
        logger.info("==========================================")
        logger.info("🔍 INICIANDO PREDICCIÓN")
        logger.info("==========================================")


        # ==================================================
        # VERIFICAR ROSTRO
        # ==================================================

        if rostro is None:

            logger.error(
                "❌ No se recibió ningún rostro"
            )

            return (
                "Neutral",
                0.0,
                None
            )


        logger.info(
            f"📷 Rostro recibido: {rostro.shape}"
        )

        logger.info(
            f"📷 Tipo: {rostro.dtype}"
        )


        # ==================================================
        # GUARDAR ROSTRO ORIGINAL
        # ==================================================

        ruta_original = os.path.join(
            DEBUG_DIR,
            "01_rostro_recibido.jpg"
        )

        cv2.imwrite(
            ruta_original,
            rostro
        )

        logger.info(
            f"📸 Rostro original guardado:\n"
            f"{ruta_original}"
        )


        # ==================================================
        # CONVERTIR A ESCALA DE GRISES
        # ==================================================

        if len(rostro.shape) == 3:

            rostro_gray = cv2.cvtColor(
                rostro,
                cv2.COLOR_BGR2GRAY
            )

        else:

            rostro_gray = rostro.copy()


        logger.info(
            f"⚫ Rostro en gris: "
            f"{rostro_gray.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO EN GRIS
        # ==================================================

        ruta_gris = os.path.join(
            DEBUG_DIR,
            "02_rostro_gris.jpg"
        )

        cv2.imwrite(
            ruta_gris,
            rostro_gray
        )


        # ==================================================
        # REDIMENSIONAR A 48x48
        #
        # IGUAL QUE EL DATASET DE ENTRENAMIENTO
        # ==================================================

        rostro_gray = cv2.resize(
            rostro_gray,
            IMG_SIZE
        )


        logger.info(
            f"📐 Rostro después de resize: "
            f"{rostro_gray.shape}"
        )


        # ==================================================
        # GUARDAR EXACTAMENTE LO QUE SE ENVÍA
        # ==================================================

        ruta_modelo = os.path.join(
            DEBUG_DIR,
            "03_rostro_para_modelo.jpg"
        )

        cv2.imwrite(
            ruta_modelo,
            rostro_gray
        )


        logger.info(
            f"📸 Rostro enviado al modelo guardado:\n"
            f"{ruta_modelo}"
        )


        # ==================================================
        # EJECUTAR MODELO LBPH
        # ==================================================

        logger.info(
            "🚀 EJECUTANDO MODELO LBPH..."
        )


        label, distancia = emotion_recognizer.predict(
            rostro_gray
        )


        logger.info(
            "✅ Modelo ejecutado correctamente"
        )


        logger.info(
            f"🏷️ Label obtenido: {label}"
        )


        logger.info(
            f"📏 Distancia LBPH: {distancia:.4f}"
        )


        # ==================================================
        # VERIFICAR LABEL
        # ==================================================

        if label < 0 or label >= len(EMOCIONES):

            logger.error(
                f"❌ Label inválido: {label}"
            )

            return (
                "Neutral",
                0.0,
                None
            )


        # ==================================================
        # OBTENER EMOCIÓN
        # ==================================================

        emocion = EMOCIONES[label]


        # ==================================================
        # CONFIANZA APROXIMADA
        #
        # LBPH devuelve DISTANCIA.
        #
        # Menor distancia = mejor coincidencia.
        #
        # NO es una probabilidad real.
        # ==================================================

        confianza = max(
            0.0,
            min(
                100.0,
                100.0 - distancia
            )
        )


        # ==================================================
        # MOSTRAR RESULTADO
        # ==================================================

        logger.info(
            f"🏆 EMOCIÓN FINAL: {emocion}"
        )

        logger.info(
            f"🏆 DISTANCIA: {distancia:.4f}"
        )

        logger.info(
            f"🏆 CONFIANZA APROXIMADA: "
            f"{confianza:.2f}%"
        )


        # ==================================================
        # GUARDAR RESULTADO
        # ==================================================

        imagen_resultado = cv2.imread(
            ruta_modelo
        )


        if imagen_resultado is not None:

            texto = (
                f"{emocion} "
                f"({confianza:.1f}%)"
            )


            cv2.putText(
                imagen_resultado,
                texto,
                (2, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )


            ruta_resultado = os.path.join(
                DEBUG_DIR,
                "04_resultado_emocion.jpg"
            )


            cv2.imwrite(
                ruta_resultado,
                imagen_resultado
            )


            logger.info(
                f"📸 Resultado guardado:\n"
                f"{ruta_resultado}"
            )


        logger.info("==========================================")
        logger.info("✅ PREDICCIÓN FINALIZADA")
        logger.info("==========================================")


        # ==================================================
        # RETORNAR
        # ==================================================

        return (
            emocion,
            confianza,
            {
                "label": int(label),
                "distancia": float(distancia)
            }
        )


    except Exception as e:

        logger.exception(
            f"❌ Error en predicción: {e}"
        )

        return (
            "Neutral",
            0.0,
            None
        )


# ==========================================================
# EJECUTAR CÁMARA
# ==========================================================

if __name__ == "__main__":

    logger.info("")
    logger.info("==========================================")
    logger.info("       🎥 INICIANDO CÁMARA")
    logger.info("==========================================")


    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        raise RuntimeError(
            "❌ No se pudo abrir la cámara"
        )


    while True:

        # ==================================================
        # LEER FRAME
        # ==================================================

        ret, frame = cap.read()


        if not ret:

            logger.error(
                "❌ No se pudo leer la cámara"
            )

            break


        # ==================================================
        # CONVERTIR FRAME A GRIS
        # ==================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ==================================================
        # DETECTAR ROSTROS
        # ==================================================

        rostros = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ==================================================
        # PROCESAR CADA ROSTRO
        # ==================================================

        for (x, y, w, h) in rostros:


            # ==============================================
            # EXTRAER ROSTRO
            # ==============================================

            rostro = frame[
                y:y+h,
                x:x+w
            ]


            # ==============================================
            # PREDECIR
            # ==============================================

            emocion, confianza, datos = predecir(
                rostro
            )


            # ==============================================
            # DIBUJAR RECTÁNGULO
            # ==============================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ==============================================
            # MOSTRAR EMOCIÓN
            # ==============================================

            texto = (
                f"{emocion} "
                f"{confianza:.1f}%"
            )


            cv2.putText(
                frame,
                texto,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ==================================================
        # MOSTRAR CÁMARA
        # ==================================================

        cv2.imshow(
            "EmotiScan - Reconocimiento de emociones",
            frame
        )


        # ==================================================
        # SALIR CON ESC
        # ==================================================

        tecla = cv2.waitKey(1) & 0xFF


        if tecla == 27:

            break


    # ======================================================
    # CERRAR CÁMARA
    # ======================================================

    cap.release()

    cv2.destroyAllWindows()


    logger.info(
        "👋 Programa finalizado"
    )