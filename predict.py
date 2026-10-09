import os
import logging
import random

import cv2
import numpy as np


# ==========================================================
# TFLITE
# ==========================================================

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
#
# ESTE ORDEN ES EXACTAMENTE EL DEL ENTRENAMIENTO
#
# 0 = Enojo
# 1 = Felicidad
# 2 = Tristeza
# 3 = Neutral
# ==========================================================

EMOCIONES = [
    "Enojo",      # Índice 0: Enojo en el entrenamiento
    "Felicidad",  # Índice 1
    "Neutral",    # Índice 2
    "Tristeza"    # Índice 3
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

    "Tristeza": [
        "Tómate un momento para descansar y respirar tranquilamente.",
        "Hablar con alguien de confianza sobre cómo te sientes puede ayudarte.",
        "Realiza una actividad que disfrutes y que te permita despejar la mente.",
        "Recuerda que está bien sentirse triste. Date tiempo para procesar lo que estás viviendo.",
        "Cuida de ti, descansa y busca compañía si sientes que necesitas apoyo."
    ],

    "Neutral": [
        "Tu expresión parece tranquila. Aprovecha este momento de calma.",
        "Puedes continuar con tus actividades manteniendo este estado de tranquilidad.",
        "Tómate unos minutos para respirar y mantener tu mente relajada.",
        "Este puede ser un buen momento para concentrarte en una tarea importante.",
        "Disfruta este momento de equilibrio y continúa cuidando tu bienestar."
    ]
}


# ==========================================================
# VARIABLES GLOBALES
# ==========================================================

interpreter = None
input_details = None
output_details = None


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

    logger.info(
        "📐 Cuantización entrada: %s",
        input_details[0].get("quantization")
    )

    logger.info(
        "📐 Cuantización salida: %s",
        output_details[0].get("quantization")
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
# PREPROCESAMIENTO RESNET50
# ==========================================================
#
# MUY IMPORTANTE:
#
# entrenar.py utiliza:
#
# from tensorflow.keras.applications.resnet50 import preprocess_input
#
# El preprocess_input de ResNet50 utiliza el modo "caffe":
#
# RGB
# BGR conceptual
# resta:
# R -> 103.939
# G -> 116.779
# B -> 123.68
#
# Aquí reproducimos ese procesamiento con NumPy
# para NO necesitar TensorFlow en producción.
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

    # ======================================================
    # CONVERTIR A RGB
    # ======================================================

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

        # OpenCV entrega BGR.
        # El entrenamiento recibe imágenes RGB.
        rostro = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

    else:

        raise ValueError(
            f"Formato de rostro no válido: {rostro.shape}"
        )

    # ======================================================
    # REDIMENSIONAR
    # ======================================================

    rostro = cv2.resize(
        rostro,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    logger.info(
        "📦 Imagen antes del preprocess: "
        "min=%.2f max=%.2f",
        rostro.min(),
        rostro.max()
    )

    # ======================================================
    # FLOAT32
    # ======================================================

    rostro = rostro.astype(
        np.float32
    )

    # ======================================================
    # PREPROCESS_INPUT DE RESNET50
    # ======================================================
    #
    # Equivalente al:
    #
    # tensorflow.keras.applications.resnet50.preprocess_input
    #
    # modo caffe.
    #
    # Primero RGB -> BGR
    # Después resta medias de ImageNet.
    # ======================================================

    rostro = rostro[..., ::-1]

    rostro[..., 0] -= 103.939
    rostro[..., 1] -= 116.779
    rostro[..., 2] -= 123.68

    # ======================================================
    # AGREGAR BATCH
    # ======================================================

    rostro = np.expand_dims(
        rostro,
        axis=0
    )

    logger.info(
        "📦 Entrada final: %s",
        rostro.shape
    )

    logger.info(
        "📊 Después preprocess ResNet50: "
        "min=%.4f max=%.4f",
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
# CONVERTIR SALIDA TFLITE
# ==========================================================

def procesar_salida(resultado):

    salida = np.asarray(
        resultado[0]
    )

    output_info = output_details[0]

    scale, zero_point = output_info.get(
        "quantization",
        (0.0, 0)
    )

    # ======================================================
    # SALIDA CUANTIZADA
    # ======================================================

    if (
        salida.dtype != np.float32
        and scale is not None
        and scale > 0
    ):

        logger.info(
            "🔄 Descuantizando salida TFLite"
        )

        salida = (
            salida.astype(np.float32)
            - zero_point
        ) * scale

    else:

        salida = salida.astype(
            np.float32
        )

    return salida


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info("=" * 60)
        logger.info("🧠 INICIANDO PREDICCIÓN")

        # ==================================================
        # CARGAR MODELO
        # ==================================================

        cargar_modelo()

        # ==================================================
        # PREPARAR ROSTRO
        # ==================================================

        imagen = preparar_rostro(
            rostro
        )

        # ==================================================
        # TIPO ESPERADO
        # ==================================================

        input_dtype = input_details[0]["dtype"]

        logger.info(
            "📌 Tipo esperado por modelo: %s",
            input_dtype
        )

        # ==================================================
        # FLOAT32
        # ==================================================

        if input_dtype == np.float32:

            imagen_final = imagen.astype(
                np.float32
            )

        # ==================================================
        # FLOAT16
        # ==================================================

        elif input_dtype == np.float16:

            imagen_final = imagen.astype(
                np.float16
            )

        # ==================================================
        # INT8 / UINT8
        # ==================================================

        elif (
            input_dtype == np.int8
            or input_dtype == np.uint8
        ):

            scale, zero_point = input_details[0].get(
                "quantization",
                (0.0, 0)
            )

            if scale is None or scale == 0:

                raise ValueError(
                    "El modelo tiene entrada cuantizada "
                    "pero no tiene una escala válida."
                )

            logger.info(
                "🔄 Cuantizando entrada: "
                "scale=%s zero_point=%s",
                scale,
                zero_point
            )

            imagen_final = (
                imagen / scale
                + zero_point
            )

            imagen_final = np.round(
                imagen_final
            )

            if input_dtype == np.int8:

                imagen_final = np.clip(
                    imagen_final,
                    -128,
                    127
                )

            else:

                imagen_final = np.clip(
                    imagen_final,
                    0,
                    255
                )

            imagen_final = imagen_final.astype(
                input_dtype
            )

        else:

            raise ValueError(
                f"Tipo de entrada TFLite no soportado: "
                f"{input_dtype}"
            )

        # ==================================================
        # LOG
        # ==================================================

        logger.info(
            "📥 Tensor enviado: "
            "shape=%s dtype=%s min=%.4f max=%.4f",
            imagen_final.shape,
            imagen_final.dtype,
            imagen_final.min(),
            imagen_final.max()
        )

        # ==================================================
        # ENVIAR TENSOR
        # ==================================================

        interpreter.set_tensor(
            input_details[0]["index"],
            imagen_final
        )

        # ==================================================
        # INFERENCIA
        # ==================================================

        interpreter.invoke()

        # ==================================================
        # OBTENER RESULTADO
        # ==================================================

        resultado = interpreter.get_tensor(
            output_details[0]["index"]
        )

        logger.info(
            "📤 Salida RAW: %s",
            resultado
        )

        # ==================================================
        # PROCESAR SALIDA
        # ==================================================

        predicciones = procesar_salida(
            resultado
        )

        logger.info(
            "📊 Predicciones procesadas: %s",
            predicciones
        )

        # ==================================================
        # VERIFICAR CLASES
        # ==================================================

        if len(predicciones) != len(EMOCIONES):

            raise ValueError(
                f"El modelo devuelve "
                f"{len(predicciones)} clases, "
                f"pero se esperan "
                f"{len(EMOCIONES)}."
            )

        # ==================================================
        # CONVERTIR A PROBABILIDADES
        # ==================================================

        suma = float(
            np.sum(predicciones)
        )

        logger.info(
            "📊 Suma salida: %.6f",
            suma
        )

        if (
            np.all(predicciones >= 0)
            and suma > 0
            and abs(suma - 1.0) < 0.05
        ):

            probabilidades = (
                predicciones / suma
            )

        else:

            logger.info(
                "🔄 Aplicando Softmax a la salida"
            )

            probabilidades = softmax(
                predicciones
            )
            

        # ==================================================
        # MOSTRAR LAS 4 EMOCIONES
        # ==================================================

        logger.info("=" * 50)
        logger.info("🎭 RESULTADO DE LAS 4 EMOCIONES")

        todas = {}

        for i, emocion_nombre in enumerate(
            EMOCIONES
        ):

            porcentaje = float(
                probabilidades[i] * 100
            )

            todas[emocion_nombre] = round(
                porcentaje,
                2
            )

            logger.info(
                "   %s: %.2f%%",
                emocion_nombre,
                porcentaje
            )

        # ==================================================
        # EMOCIÓN PRINCIPAL
        # ==================================================

        indice = int(
            np.argmax(
                probabilidades
            )
        )

        emocion = EMOCIONES[
            indice
        ]

        confianza = float(
            probabilidades[indice] * 100
        )

        logger.info(
            "🏆 PRINCIPAL: %s (%.2f%%)",
            emocion,
            confianza
        )

        # ==================================================
        # CONSEJO
        # ==================================================

        consejo = obtener_consejo(
            emocion
        )

        logger.info(
            "💡 CONSEJO: %s",
            consejo["mensaje"]
        )

        logger.info("=" * 60)

        # ==================================================
        # RETORNAR
        # ==================================================

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
# ==========================================================
# PRUEBA DESDE CONSOLA
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("\n❌ Debes indicar una imagen.")
        print("Ejemplo:")
        print("python predict.py enojo.jpg")
        sys.exit(1)

    ruta_imagen = sys.argv[1]

    print("\n" + "=" * 60)
    print("🧪 PRUEBA DE PREDICCIÓN")
    print("=" * 60)

    print(f"📷 Imagen: {ruta_imagen}")

    if not os.path.exists(ruta_imagen):

        print(f"❌ No existe la imagen: {ruta_imagen}")
        sys.exit(1)

    rostro = cv2.imread(
        ruta_imagen,
        cv2.IMREAD_COLOR
    )

    if rostro is None:

        print("❌ No se pudo leer la imagen.")
        sys.exit(1)

    print(
        f"📐 Tamaño de imagen: {rostro.shape}"
    )

    try:

        emocion, confianza, todas, consejo = predecir(
            rostro
        )

        print("\n" + "=" * 60)
        print("🎭 RESULTADO FINAL")
        print("=" * 60)

        for nombre, porcentaje in todas.items():

            print(
                f"{nombre}: {porcentaje:.2f}%"
            )

        print("-" * 60)

        print(
            f"🏆 PRINCIPAL: {emocion}"
        )

        print(
            f"📊 CONFIANZA: {confianza:.2f}%"
        )

        print(
            f"💡 CONSEJO: {consejo['mensaje']}"
        )

        print("=" * 60)

    except Exception as e:

        print(
            f"\n❌ Error durante la prueba: {e}"
        )