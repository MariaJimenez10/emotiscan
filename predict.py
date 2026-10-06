import os
import logging
import random

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Usando tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✅ Usando TensorFlow Lite")
# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# DIRECTORIO DEL PROYECTO
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ==========================================================
# MODELO
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)

IMG_SIZE = 224


# ==========================================================
# EMOCIONES
# ==========================================================

# ESTE ORDEN DEBE COINCIDIR CON EL ENTRENAMIENTO

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]


# ==========================================================
# CONSEJOS
# ==========================================================

CONSEJOS = {

    "Enojo": [
        "Haz una pausa y respira profundamente antes de reaccionar.",
        "Aléjate unos minutos de la situación que te está molestando.",
        "Intenta identificar qué está causando tu enojo y piensa antes de actuar.",
        "Respira lentamente, relaja tu cuerpo y date un momento para recuperar la calma.",
        "Hablar sobre lo que sientes cuando estés más tranquilo puede ayudarte a manejar mejor la situación."
    ],

    "Felicidad": [
        "¡Excelente! Disfruta este momento y comparte tu energía positiva con los demás.",
        "Aprovecha este estado de ánimo para realizar una actividad que disfrutes.",
        "Sonríe, disfruta el momento y guarda este recuerdo positivo.",
        "Compartir tu alegría con las personas que aprecias puede hacer que este momento sea aún mejor.",
        "Utiliza esta energía positiva para avanzar en algo que te haga sentir orgulloso."
    ],

    "Neutral": [
        "Tu expresión parece tranquila. Aprovecha este momento de calma.",
        "Puedes continuar con tus actividades manteniendo este estado de tranquilidad.",
        "Tómate unos minutos para respirar y mantener tu mente relajada.",
        "Este puede ser un buen momento para concentrarte en una tarea importante.",
        "Disfruta este momento de equilibrio y continúa cuidando tu bienestar."
    ],

    "Tristeza": [
        "Tómate un momento para descansar y respirar tranquilamente.",
        "Hablar con alguien de confianza sobre cómo te sientes puede ayudarte.",
        "Realiza una actividad que disfrutes y que te permita despejar la mente.",
        "Recuerda que está bien sentirse triste. Date tiempo para procesar lo que estás viviendo.",
        "Cuida de ti, descansa y busca compañía si sientes que necesitas apoyo."
    ]
}


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

interpreter = None
input_details = None
output_details = None


# ==========================================================
# PREPROCESAMIENTO RESNET50
# ==========================================================

def preprocess_resnet50(imagen):
    """
    Equivalente al preprocess_input de
    tensorflow.keras.applications.resnet50

    ResNet50 espera:
        RGB
        float32
        valores transformados de 0-255

    El procesamiento original de Keras ResNet50
    convierte RGB -> BGR y resta:
        [103.939, 116.779, 123.68]
    """

    imagen = imagen.astype(
        np.float32
    )

    # RGB -> BGR
    imagen = imagen[:, :, ::-1]

    # Restar valores de media de ResNet50
    imagen[:, :, 0] -= 103.939
    imagen[:, :, 1] -= 116.779
    imagen[:, :, 2] -= 123.680

    return imagen


# ==========================================================
# CARGAR MODELO
# ==========================================================

def cargar_modelo():

    global interpreter
    global input_details
    global output_details

    if interpreter is not None:
        return interpreter

    logger.info("=" * 60)
    logger.info("🧠 CARGANDO MODELO RESNET50 TFLITE")
    logger.info("=" * 60)

    logger.info(
        "📁 Modelo: %s",
        MODEL_PATH
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    # ======================================================
    # TFLITE
    # ======================================================

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info("✅ MODELO CARGADO")

    logger.info(
        "📥 Entrada: %s",
        input_details[0]["shape"]
    )

    logger.info(
        "📥 Tipo entrada: %s",
        input_details[0]["dtype"]
    )

    logger.info(
        "📤 Salida: %s",
        output_details[0]["shape"]
    )

    logger.info(
        "📤 Tipo salida: %s",
        output_details[0]["dtype"]
    )

    logger.info("=" * 60)

    return interpreter


# ==========================================================
# OBTENER CONSEJO
# ==========================================================

def obtener_consejo(emocion):

    mensajes = CONSEJOS.get(
        emocion,
        [
            "Tómate un momento para respirar y cuidar de ti."
        ]
    )

    return {
        "titulo": f"Consejo para {emocion}",
        "mensaje": random.choice(mensajes)
    }


# ==========================================================
# PREPARAR ROSTRO
# ==========================================================

def preparar_rostro(rostro):

    if rostro is None:

        raise ValueError(
            "No se recibió ningún rostro."
        )

    if rostro.size == 0:

        raise ValueError(
            "El rostro recibido está vacío."
        )

    logger.info(
        "📷 Rostro original: %s",
        rostro.shape
    )

    # ------------------------------------------------------
    # Convertir a RGB
    # ------------------------------------------------------

    if len(rostro.shape) == 2:

        rostro = cv2.cvtColor(
            rostro,
            cv2.COLOR_GRAY2RGB
        )

    elif (
        len(rostro.shape) == 3
        and rostro.shape[2] == 1
    ):

        rostro = cv2.cvtColor(
            rostro,
            cv2.COLOR_GRAY2RGB
        )

    elif (
        len(rostro.shape) == 3
        and rostro.shape[2] == 3
    ):

        # OpenCV trabaja en BGR
        # El modelo fue entrenado con RGB
        rostro = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

    else:

        raise ValueError(
            f"Formato de rostro no válido: {rostro.shape}"
        )

    # ------------------------------------------------------
    # Redimensionar
    # ------------------------------------------------------

    rostro = cv2.resize(
        rostro,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    logger.info(
        "📦 Antes preprocess ResNet50: min=%.2f, max=%.2f",
        rostro.min(),
        rostro.max()
    )

    # ------------------------------------------------------
    # PREPROCESAMIENTO RESNET50
    # ------------------------------------------------------

    rostro = preprocess_resnet50(
        rostro
    )

    # ------------------------------------------------------
    # Agregar batch
    # ------------------------------------------------------

    rostro = np.expand_dims(
        rostro,
        axis=0
    )

    logger.info(
        "📦 Entrada final: %s",
        rostro.shape
    )

    logger.info(
        "📊 Después preprocess ResNet50: min=%.2f, max=%.2f",
        rostro.min(),
        rostro.max()
    )

    return rostro


# ==========================================================
# SOFTMAX
# ==========================================================

def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x - np.max(x)

    exp_x = np.exp(x)

    return (
        exp_x /
        np.sum(exp_x)
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info("=" * 60)
        logger.info("🧠 INICIANDO PREDICCIÓN")

        # --------------------------------------------------
        # Cargar modelo
        # --------------------------------------------------

        cargar_modelo()

        # --------------------------------------------------
        # Preparar rostro
        # --------------------------------------------------

        imagen = preparar_rostro(
            rostro
        )

        # --------------------------------------------------
        # Verificar tipo esperado
        # --------------------------------------------------

        input_dtype = input_details[0]["dtype"]

        if imagen.dtype != input_dtype:

            imagen = imagen.astype(
                input_dtype
            )

        logger.info(
            "📥 Tensor enviado: shape=%s dtype=%s",
            imagen.shape,
            imagen.dtype
        )

        # --------------------------------------------------
        # Enviar tensor
        # --------------------------------------------------

        interpreter.set_tensor(
            input_details[0]["index"],
            imagen
        )

        # --------------------------------------------------
        # Inferencia
        # --------------------------------------------------

        interpreter.invoke()

        # --------------------------------------------------
        # Obtener resultado
        # --------------------------------------------------

        resultado = interpreter.get_tensor(
            output_details[0]["index"]
        )

        predicciones = np.asarray(
            resultado[0],
            dtype=np.float32
        )

        logger.info(
            "📤 Salida RAW: %s",
            resultado
        )

        logger.info(
            "📊 Salida final: %s",
            predicciones
        )

        # --------------------------------------------------
        # Verificar clases
        # --------------------------------------------------

        if len(predicciones) != len(EMOCIONES):

            raise ValueError(
                f"El modelo devuelve "
                f"{len(predicciones)} clases, "
                f"pero se esperan "
                f"{len(EMOCIONES)}."
            )

        # --------------------------------------------------
        # Normalizar
        # --------------------------------------------------

        suma = float(
            np.sum(predicciones)
        )

        logger.info(
            "📊 Suma probabilidades: %.6f",
            suma
        )

        if (
            np.any(predicciones < 0)
            or abs(suma - 1.0) > 0.05
        ):

            predicciones = softmax(
                predicciones
            )

        elif suma > 0:

            predicciones = (
                predicciones / suma
            )

        # --------------------------------------------------
        # Mostrar resultados
        # --------------------------------------------------

        logger.info("=" * 50)
        logger.info("🎭 RESULTADO")

        todas = {}

        for i, emocion in enumerate(
            EMOCIONES
        ):

            porcentaje = float(
                predicciones[i] * 100
            )

            todas[emocion] = round(
                porcentaje,
                2
            )

            logger.info(
                "   %s: %.2f%%",
                emocion,
                porcentaje
            )

        # --------------------------------------------------
        # Emoción principal
        # --------------------------------------------------

        indice = int(
            np.argmax(predicciones)
        )

        emocion = EMOCIONES[
            indice
        ]

        confianza = float(
            predicciones[indice] * 100
        )

        logger.info(
            "🏆 PRINCIPAL: %s (%.2f%%)",
            emocion,
            confianza
        )

        logger.info("=" * 50)

        # --------------------------------------------------
        # CONSEJO
        # --------------------------------------------------

        consejo = obtener_consejo(
            emocion
        )

        logger.info(
            "💡 CONSEJO: %s",
            consejo["mensaje"]
        )

        logger.info("=" * 60)

        # --------------------------------------------------
        # DEVOLVER LOS 4 VALORES
        # --------------------------------------------------

        return (
            emocion,
            confianza,
            todas,
            consejo
        )

    except Exception as e:

        logger.exception(
            "❌ ERROR EN PREDICCIÓN: %s",
            e
        )

        raise
