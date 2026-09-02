import os
import cv2
import logging
import numpy as np
import tensorflow as tf

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)

# IMPORTANTE:
# El orden debe coincidir con el orden usado durante
# el entrenamiento del modelo.
EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# ==========================================================
# VARIABLES DEL MODELO
# ==========================================================

interpreter = None
input_details = None
output_details = None


# ==========================================================
# CARGAR MODELO TFLITE
# ==========================================================

def cargar_modelo():

    global interpreter
    global input_details
    global output_details

    if interpreter is not None:
        return

    logger.info("======================================")
    logger.info("🧠 CARGANDO MODELO TFLITE")
    logger.info("📁 Modelo: %s", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    logger.info(
        "📦 Tamaño del modelo: %.2f KB",
        os.path.getsize(MODEL_PATH) / 1024
    )

    # Crear intérprete TFLite
    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH,
        num_threads=1
    )

    logger.info("⚙️ Asignando tensores...")

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("✅ Modelo TFLite cargado correctamente")

    logger.info(
        "📥 Entrada: shape=%s dtype=%s",
        input_details[0]["shape"],
        input_details[0]["dtype"]
    )

    logger.info(
        "📤 Salida: shape=%s dtype=%s",
        output_details[0]["shape"],
        output_details[0]["dtype"]
    )

    logger.info("======================================")


# ==========================================================
# PREPARAR ROSTRO
# ==========================================================

def preparar_rostro(rostro):

    if rostro is None:
        raise ValueError("El rostro recibido es None")

    logger.info(
        "👤 Preparando rostro: shape=%s dtype=%s",
        rostro.shape,
        rostro.dtype
    )

    # ------------------------------------------------------
    # Obtener tamaño esperado por el modelo
    # ------------------------------------------------------

    input_shape = input_details[0]["shape"]

    alto = int(input_shape[1])
    ancho = int(input_shape[2])

    logger.info(
        "📐 Tamaño requerido por el modelo: %sx%s",
        ancho,
        alto
    )

    # ------------------------------------------------------
    # Convertir BGR → RGB
    # ------------------------------------------------------

    if len(rostro.shape) == 3:

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

    else:

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_GRAY2RGB
        )

    # ------------------------------------------------------
    # Redimensionar
    # ------------------------------------------------------

    rostro_rgb = cv2.resize(
        rostro_rgb,
        (ancho, alto)
    )

    # ------------------------------------------------------
    # Convertir a float32 o uint8
    # ------------------------------------------------------

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:

        rostro_rgb = rostro_rgb.astype(np.float32)

        # Normalización típica para modelos entrenados
        # con imágenes entre 0 y 1.
        rostro_rgb = rostro_rgb / 255.0

    elif input_dtype == np.uint8:

        rostro_rgb = rostro_rgb.astype(np.uint8)

    else:

        logger.warning(
            "⚠️ Tipo de entrada no esperado: %s",
            input_dtype
        )

        rostro_rgb = rostro_rgb.astype(input_dtype)

    # ------------------------------------------------------
    # Agregar dimensión batch
    # ------------------------------------------------------

    rostro_rgb = np.expand_dims(
        rostro_rgb,
        axis=0
    )

    logger.info(
        "✅ Rostro preparado: shape=%s dtype=%s",
        rostro_rgb.shape,
        rostro_rgb.dtype
    )

    return rostro_rgb


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info("======================================")
        logger.info("🎯 INICIANDO PREDICCIÓN")

        # Cargar modelo
        cargar_modelo()

        if rostro is None:

            logger.warning(
                "❌ No se recibió rostro"
            )

            return (
                "Neutral",
                0.0,
                {}
            )

        # --------------------------------------------------
        # Preparar rostro
        # --------------------------------------------------

        imagen = preparar_rostro(rostro)

        # --------------------------------------------------
        # Pasar imagen al modelo
        # --------------------------------------------------

        logger.info("📤 Enviando rostro al modelo...")

        interpreter.set_tensor(
            input_details[0]["index"],
            imagen
        )

        # --------------------------------------------------
        # Ejecutar modelo
        # --------------------------------------------------

        logger.info("🧠 Ejecutando inferencia...")

        interpreter.invoke()

        logger.info("✅ Inferencia terminada")

        # --------------------------------------------------
        # Obtener resultado
        # --------------------------------------------------

        resultado = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicciones = resultado[0]

        logger.info(
            "📊 Predicciones: %s",
            predicciones
        )

        # --------------------------------------------------
        # Obtener clase con mayor probabilidad
        # --------------------------------------------------

        indice = int(
            np.argmax(predicciones)
        )

        confianza = float(
            predicciones[indice]
        )

        # --------------------------------------------------
        # Convertir a porcentaje si está entre 0 y 1
        # --------------------------------------------------

        if confianza <= 1.0:

            confianza_porcentaje = (
                confianza * 100
            )

        else:

            confianza_porcentaje = confianza

        # --------------------------------------------------
        # Validar índice
        # --------------------------------------------------

        if indice < 0 or indice >= len(EMOCIONES):

            logger.warning(
                "⚠️ Índice inválido: %s",
                indice
            )

            return (
                "Neutral",
                0.0,
                {
                    "indice": indice,
                    "predicciones": predicciones.tolist()
                }
            )

        emocion = EMOCIONES[indice]

        # --------------------------------------------------
        # Crear diccionario de resultados
        # --------------------------------------------------

        resultados = {}

        for i, emocion_nombre in enumerate(EMOCIONES):

            if i < len(predicciones):

                valor = float(
                    predicciones[i]
                )

                if valor <= 1.0:
                    valor *= 100

                resultados[emocion_nombre] = round(
                    valor,
                    2
                )

        logger.info(
            "🎭 EMOCIÓN: %s",
            emocion
        )

        logger.info(
            "📈 CONFIANZA: %.2f%%",
            confianza_porcentaje
        )

        logger.info(
            "📊 TODAS LAS EMOCIONES: %s",
            resultados
        )

        logger.info("======================================")

        return (
            emocion,
            confianza_porcentaje,
            resultados
        )

    except Exception as e:

        logger.exception(
            "❌ Error en predicción: %s",
            e
        )

        return (
            "Neutral",
            0.0,
            {
                "error": str(e)
            }
        )