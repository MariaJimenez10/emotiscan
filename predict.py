import os
import cv2
import logging
import numpy as np

from tflite_runtime.interpreter import Interpreter


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

# ==========================================================
# EMOCIONES
# ==========================================================
#
# IMPORTANTE:
# Este orden DEBE coincidir con el orden utilizado
# durante el entrenamiento del modelo.
#

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

    # Si ya está cargado, no volver a cargarlo
    if interpreter is not None:
        return

    logger.info("======================================")
    logger.info("🧠 CARGANDO MODELO RESNET50 TFLITE")
    logger.info("📁 Modelo: %s", MODEL_PATH)

    # ------------------------------------------------------
    # Verificar que exista
    # ------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    # ------------------------------------------------------
    # Tamaño del archivo
    # ------------------------------------------------------

    tamanio_kb = os.path.getsize(MODEL_PATH) / 1024

    logger.info(
        "📦 Tamaño del modelo: %.2f KB",
        tamanio_kb
    )

    # ------------------------------------------------------
    # Crear intérprete
    # ------------------------------------------------------

    logger.info("⚙️ Creando intérprete TFLite...")

    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=1
    )

    # ------------------------------------------------------
    # Reservar tensores
    # ------------------------------------------------------

    logger.info("⚙️ Asignando tensores...")

    interpreter.allocate_tensors()

    # ------------------------------------------------------
    # Obtener información de entrada y salida
    # ------------------------------------------------------

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("✅ MODELO CARGADO CORRECTAMENTE")

    # Entrada
    logger.info(
        "📥 Entrada:"
    )

    logger.info(
        "   Shape: %s",
        input_details[0]["shape"]
    )

    logger.info(
        "   Tipo: %s",
        input_details[0]["dtype"]
    )

    # Salida
    logger.info(
        "📤 Salida:"
    )

    logger.info(
        "   Shape: %s",
        output_details[0]["shape"]
    )

    logger.info(
        "   Tipo: %s",
        output_details[0]["dtype"]
    )

    logger.info("======================================")


# ==========================================================
# PREPARAR ROSTRO
# ==========================================================

def preparar_rostro(rostro):

    if rostro is None:

        raise ValueError(
            "El rostro recibido es None"
        )

    logger.info(
        "👤 Rostro recibido: shape=%s dtype=%s",
        rostro.shape,
        rostro.dtype
    )

    # ------------------------------------------------------
    # Obtener tamaño que necesita el modelo
    # ------------------------------------------------------

    input_shape = input_details[0]["shape"]

    alto = int(input_shape[1])
    ancho = int(input_shape[2])

    logger.info(
        "📐 Tamaño requerido: %sx%s",
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

    elif len(rostro.shape) == 2:

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_GRAY2RGB
        )

    else:

        raise ValueError(
            f"Formato de rostro no válido: {rostro.shape}"
        )

    # ------------------------------------------------------
    # Redimensionar
    # ------------------------------------------------------

    rostro_rgb = cv2.resize(
        rostro_rgb,
        (ancho, alto),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------
    # Obtener tipo de entrada
    # ------------------------------------------------------

    input_dtype = input_details[0]["dtype"]

    logger.info(
        "🔢 Tipo de entrada esperado: %s",
        input_dtype
    )

    # ------------------------------------------------------
    # Preparar según el tipo
    # ------------------------------------------------------

    if input_dtype == np.float32:

        rostro_rgb = rostro_rgb.astype(
            np.float32
        )

        # Normalización 0-1
        rostro_rgb = rostro_rgb / 255.0

    elif input_dtype == np.uint8:

        rostro_rgb = rostro_rgb.astype(
            np.uint8
        )

    elif input_dtype == np.int8:

        rostro_rgb = rostro_rgb.astype(
            np.int8
        )

    else:

        logger.warning(
            "⚠️ Tipo de entrada no contemplado: %s",
            input_dtype
        )

        rostro_rgb = rostro_rgb.astype(
            input_dtype
        )

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

        # --------------------------------------------------
        # Cargar modelo
        # --------------------------------------------------

        cargar_modelo()

        # --------------------------------------------------
        # Verificar rostro
        # --------------------------------------------------

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

        imagen = preparar_rostro(
            rostro
        )

        # --------------------------------------------------
        # Enviar imagen al modelo
        # --------------------------------------------------

        logger.info(
            "📤 Enviando rostro al modelo..."
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            imagen
        )

        # --------------------------------------------------
        # Ejecutar inferencia
        # --------------------------------------------------

        logger.info(
            "🧠 Ejecutando ResNet50..."
        )

        interpreter.invoke()

        logger.info(
            "✅ Inferencia terminada"
        )

        # --------------------------------------------------
        # Obtener salida
        # --------------------------------------------------

        resultado = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicciones = resultado[0]

        logger.info(
            "📊 Predicciones crudas: %s",
            predicciones
        )

        # --------------------------------------------------
        # Verificar cantidad de clases
        # --------------------------------------------------

        if len(predicciones) != len(EMOCIONES):

            logger.error(
                "❌ El modelo devuelve %d clases, "
                "pero EMOCIONES tiene %d",
                len(predicciones),
                len(EMOCIONES)
            )

            return (
                "Neutral",
                0.0,
                {
                    "predicciones": predicciones.tolist()
                }
            )

        # --------------------------------------------------
        # Convertir predicciones a probabilidades
        # --------------------------------------------------

        predicciones = np.array(
            predicciones,
            dtype=np.float32
        )

        # Si la salida no parece estar normalizada,
        # aplicar Softmax.

        suma = np.sum(predicciones)

        if (
            np.any(predicciones < 0)
            or suma < 0.9
            or suma > 1.1
        ):

            exp_values = np.exp(
                predicciones -
                np.max(predicciones)
            )

            predicciones = (
                exp_values /
                np.sum(exp_values)
            )

        # --------------------------------------------------
        # Obtener emoción principal
        # --------------------------------------------------

        indice = int(
            np.argmax(predicciones)
        )

        confianza = float(
            predicciones[indice]
        )

        confianza_porcentaje = (
            confianza * 100
        )

        # --------------------------------------------------
        # Nombre de la emoción
        # --------------------------------------------------

        emocion = EMOCIONES[indice]

        # --------------------------------------------------
        # Crear resultados de todas las emociones
        # --------------------------------------------------

        resultados = {}

        for i, nombre in enumerate(EMOCIONES):

            porcentaje = float(
                predicciones[i] * 100
            )

            resultados[nombre] = round(
                porcentaje,
                2
            )

        # --------------------------------------------------
        # Logs
        # --------------------------------------------------

        logger.info(
            "🎭 EMOCIÓN DETECTADA: %s",
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

        # --------------------------------------------------
        # Retornar resultado
        # --------------------------------------------------

        return (
            emocion,
            confianza_porcentaje,
            resultados
        )

    except Exception as e:

        logger.exception(
            "❌ ERROR EN PREDICCIÓN: %s",
            e
        )

        return (
            "Neutral",
            0.0,
            {
                "error": str(e)
            }
        )