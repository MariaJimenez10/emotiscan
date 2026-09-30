import os
import logging
import cv2
import numpy as np
import time
import random

from tflite_runtime.interpreter import Interpreter


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# DIRECTORIO DEL PROYECTO
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================================
# MODELO TFLITE
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)


# ==========================================================
# CONFIGURACIÓN DE IMAGEN
# ==========================================================

IMG_SIZE = 224


# ==========================================================
# CARPETA PARA GUARDAR ROSTROS
# ==========================================================

ROSTROS_DIR = os.path.join(
    BASE_DIR,
    "rostros_detectados"
)

os.makedirs(
    ROSTROS_DIR,
    exist_ok=True
)


# ==========================================================
# INTERVALO DE GUARDADO
# ==========================================================

INTERVALO_GUARDADO = 2.0


# ==========================================================
# EMOCIONES
# ==========================================================

# IMPORTANTE:
#
# Este orden DEBE coincidir exactamente con el orden
# utilizado durante el entrenamiento del modelo.
#
# 0 -> Enojo
# 1 -> Felicidad
# 2 -> Neutral
# 3 -> Tristeza
#
# ==========================================================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
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
# VARIABLES GLOBALES DEL MODELO
# ==========================================================

interpreter = None

input_details = None

output_details = None


# ==========================================================
# VARIABLES DEL CONSEJO
# ==========================================================

ultima_emocion_consejo = None

consejo_actual = None


# ==========================================================
# CARGAR MODELO TFLITE
# ==========================================================

def cargar_modelo():

    global interpreter
    global input_details
    global output_details

    # ------------------------------------------------------
    # Si ya está cargado
    # ------------------------------------------------------

    if interpreter is not None:

        return interpreter

    logger.info(
        "=" * 60
    )

    logger.info(
        "🧠 CARGANDO MODELO RESNET50 TFLITE"
    )

    logger.info(
        "📁 Modelo: %s",
        MODEL_PATH
    )

    logger.info(
        "=" * 60
    )

    # ------------------------------------------------------
    # Verificar existencia
    # ------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No se encontró el modelo TFLite:\n{MODEL_PATH}"
        )

    # ------------------------------------------------------
    # Tamaño del modelo
    # ------------------------------------------------------

    tamanio_mb = (
        os.path.getsize(MODEL_PATH)
        /
        (1024 * 1024)
    )

    logger.info(
        "📦 Tamaño del modelo: %.2f MB",
        tamanio_mb
    )

    # ------------------------------------------------------
    # Crear intérprete
    # ------------------------------------------------------

    logger.info(
        "⚙️ Creando intérprete TFLite..."
    )

    interpreter = Interpreter(
        model_path=MODEL_PATH,
        num_threads=1
    )

    # ------------------------------------------------------
    # Reservar tensores
    # ------------------------------------------------------

    logger.info(
        "⚙️ Asignando tensores..."
    )

    interpreter.allocate_tensors()

    # ------------------------------------------------------
    # Obtener información
    # ------------------------------------------------------

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    # ------------------------------------------------------
    # Mostrar entrada
    # ------------------------------------------------------

    logger.info(
        "📥 Entrada del modelo:"
    )

    logger.info(
        "   Shape: %s",
        input_details[0]["shape"]
    )

    logger.info(
        "   Tipo: %s",
        input_details[0]["dtype"]
    )

    # ------------------------------------------------------
    # Mostrar salida
    # ------------------------------------------------------

    logger.info(
        "📤 Salida del modelo:"
    )

    logger.info(
        "   Shape: %s",
        output_details[0]["shape"]
    )

    logger.info(
        "   Tipo: %s",
        output_details[0]["dtype"]
    )

    # ------------------------------------------------------
    # Mostrar clases
    # ------------------------------------------------------

    logger.info(
        "🎭 Orden de clases:"
    )

    for i, emocion in enumerate(EMOCIONES):

        logger.info(
            "   Índice %d → %s",
            i,
            emocion
        )

    logger.info(
        "✅ MODELO TFLITE CARGADO CORRECTAMENTE"
    )

    logger.info(
        "=" * 60
    )

    return interpreter


# ==========================================================
# OBTENER CONSEJO
# ==========================================================

def obtener_consejo(emocion):

    global ultima_emocion_consejo
    global consejo_actual

    # ------------------------------------------------------
    # Cambió la emoción
    # ------------------------------------------------------

    if emocion != ultima_emocion_consejo:

        mensajes = CONSEJOS.get(
            emocion,
            [
                "Tómate un momento para respirar y cuidar de ti."
            ]
        )

        mensaje = random.choice(
            mensajes
        )

        consejo_actual = {
            "titulo": f"Consejo para {emocion}",
            "mensaje": mensaje
        }

        ultima_emocion_consejo = emocion

    return consejo_actual


# ==========================================================
# PREPARAR ROSTRO
# ==========================================================

def preparar_rostro(rostro):

    if rostro is None:

        raise ValueError(
            "El rostro recibido es None."
        )

    if not isinstance(
        rostro,
        np.ndarray
    ):

        raise ValueError(
            "El rostro debe ser un numpy.ndarray."
        )

    if rostro.size == 0:

        raise ValueError(
            "El rostro está vacío."
        )

    logger.info(
        "👤 Rostro recibido: shape=%s dtype=%s",
        rostro.shape,
        rostro.dtype
    )

    # ======================================================
    # CONVERTIR A RGB
    # ======================================================

    if len(rostro.shape) == 2:

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_GRAY2RGB
        )

    elif (
        len(rostro.shape) == 3
        and rostro.shape[2] == 1
    ):

        rostro_gris = rostro[:, :, 0]

        rostro_rgb = cv2.cvtColor(
            rostro_gris,
            cv2.COLOR_GRAY2RGB
        )

    elif (
        len(rostro.shape) == 3
        and rostro.shape[2] == 3
    ):

        # OpenCV trabaja en BGR.
        # El modelo fue entrenado con RGB.

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

    else:

        raise ValueError(
            f"Formato de rostro no válido: {rostro.shape}"
        )

    # ======================================================
    # OBTENER TAMAÑO DEL MODELO
    # ======================================================

    input_shape = input_details[0]["shape"]

    alto = int(
        input_shape[1]
    )

    ancho = int(
        input_shape[2]
    )

    logger.info(
        "📐 Tamaño requerido por modelo: %sx%s",
        ancho,
        alto
    )

    # ======================================================
    # REDIMENSIONAR
    # ======================================================

    rostro_rgb = cv2.resize(
        rostro_rgb,
        (ancho, alto),
        interpolation=cv2.INTER_AREA
    )

    # ======================================================
    # OBTENER TIPO DE ENTRADA
    # ======================================================

    input_dtype = input_details[0]["dtype"]

    logger.info(
        "🔢 Tipo de entrada esperado: %s",
        input_dtype
    )

    # ======================================================
    # PREPROCESAMIENTO
    # ======================================================

    if input_dtype == np.float32:

        # --------------------------------------------------
        # IMPORTANTE
        #
        # ResNet50 normalmente utiliza la transformación:
        #
        # RGB 0-255
        #        ↓
        # preprocess_input
        #
        # equivalente a la normalización utilizada por
        # ResNet50 de Keras.
        #
        # --------------------------------------------------

        rostro_rgb = rostro_rgb.astype(
            np.float32
        )

        rostro_rgb = (
            rostro_rgb / 127.5
        ) - 1.0

    elif input_dtype == np.uint8:

        rostro_rgb = np.clip(
            rostro_rgb,
            0,
            255
        ).astype(
            np.uint8
        )

    elif input_dtype == np.int8:

        # --------------------------------------------------
        # Para modelos int8 utilizamos cuantización
        # si el modelo proporciona escala y zero point.
        # --------------------------------------------------

        scale, zero_point = (
            input_details[0].get(
                "quantization",
                (0.0, 0)
            )
        )

        if scale != 0:

            rostro_rgb = (
                rostro_rgb.astype(
                    np.float32
                )
                / scale
            ) + zero_point

            rostro_rgb = np.clip(
                rostro_rgb,
                -128,
                127
            ).astype(
                np.int8
            )

        else:

            rostro_rgb = np.clip(
                rostro_rgb,
                -128,
                127
            ).astype(
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

    # ======================================================
    # AGREGAR BATCH
    # ======================================================

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
# SOFTMAX
# ==========================================================

def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x - np.max(x)

    exp_x = np.exp(
        x
    )

    suma = np.sum(
        exp_x
    )

    if suma == 0:

        return np.ones_like(
            exp_x
        ) / len(exp_x)

    return (
        exp_x / suma
    )


# ==========================================================
# CONVERTIR SALIDA A PROBABILIDADES
# ==========================================================

def convertir_probabilidades(predicciones):

    predicciones = np.asarray(
        predicciones,
        dtype=np.float32
    )

    # ------------------------------------------------------
    # Aplanar
    # ------------------------------------------------------

    predicciones = np.squeeze(
        predicciones
    )

    # ------------------------------------------------------
    # Verificar cantidad
    # ------------------------------------------------------

    if len(predicciones) != len(
        EMOCIONES
    ):

        raise ValueError(
            f"El modelo devuelve "
            f"{len(predicciones)} clases, "
            f"pero EMOCIONES tiene "
            f"{len(EMOCIONES)}."
        )

    # ------------------------------------------------------
    # Verificar si ya son probabilidades
    # ------------------------------------------------------

    suma = float(
        np.sum(predicciones)
    )

    son_probabilidades = (
        np.all(predicciones >= 0)
        and
        np.all(predicciones <= 1)
        and
        abs(suma - 1.0) < 0.05
    )

    if son_probabilidades:

        if suma > 0:

            predicciones = (
                predicciones / suma
            )

    else:

        predicciones = softmax(
            predicciones
        )

    return predicciones


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info(
            "=========================================="
        )

        logger.info(
            "🎯 INICIANDO PREDICCIÓN"
        )

        # ==================================================
        # CARGAR MODELO
        # ==================================================

        cargar_modelo()

        # ==================================================
        # VALIDAR ROSTRO
        # ==================================================

        if rostro is None:

            logger.warning(
                "⚠️ No se recibió ningún rostro"
            )

            return (
                "Neutral",
                0.0,
                {
                    "Enojo": 0.0,
                    "Felicidad": 0.0,
                    "Neutral": 100.0,
                    "Tristeza": 0.0
                },
                obtener_consejo("Neutral")
            )

        # ==================================================
        # PREPARAR ROSTRO
        # ==================================================

        imagen = preparar_rostro(
            rostro
        )

        # ==================================================
        # ENVIAR AL MODELO
        # ==================================================

        logger.info(
            "📤 Enviando rostro al modelo..."
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            imagen
        )

        # ==================================================
        # EJECUTAR INFERENCIA
        # ==================================================

        logger.info(
            "🧠 Ejecutando ResNet50 TFLite..."
        )

        interpreter.invoke()

        logger.info(
            "✅ Inferencia terminada"
        )

        # ==================================================
        # OBTENER SALIDA
        # ==================================================

        resultado = interpreter.get_tensor(
            output_details[0]["index"]
        )

        logger.info(
            "📊 Salida cruda: %s",
            resultado
        )

        # ==================================================
        # PROBABILIDADES
        # ==================================================

        probabilidades = (
            convertir_probabilidades(
                resultado
            )
        )

        # ==================================================
        # MOSTRAR PROBABILIDADES
        # ==================================================

        logger.info(
            "=========================================="
        )

        logger.info(
            "📊 PROBABILIDADES"
        )

        logger.info(
            "=========================================="
        )

        for i, nombre in enumerate(
            EMOCIONES
        ):

            logger.info(
                "%-12s: %6.2f%%",
                nombre,
                probabilidades[i] * 100
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
            probabilidades[indice]
            * 100
        )

        # ==================================================
        # TODAS LAS EMOCIONES
        # ==================================================

        todas = {}

        for i, nombre in enumerate(
            EMOCIONES
        ):

            todas[nombre] = round(
                float(
                    probabilidades[i]
                    * 100
                ),
                2
            )

        # ==================================================
        # CONSEJO
        # ==================================================

        consejo = obtener_consejo(
            emocion
        )

        # ==================================================
        # LOGS
        # ==================================================

        logger.info(
            "------------------------------------------"
        )

        logger.info(
            "🎯 EMOCIÓN: %s",
            emocion
        )

        logger.info(
            "📈 CONFIANZA: %.2f%%",
            confianza
        )

        logger.info(
            "📊 TODAS: %s",
            todas
        )

        logger.info(
            "=========================================="
        )

        # ==================================================
        # RETORNAR RESULTADO
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

        return (
            "Neutral",
            0.0,
            {
                "Enojo": 0.0,
                "Felicidad": 0.0,
                "Neutral": 0.0,
                "Tristeza": 0.0
            },
            {
                "titulo": "Sin resultado",
                "mensaje": "No fue posible analizar el rostro."
            }
        )


# ==========================================================
# GUARDAR ROSTRO
# ==========================================================

def guardar_rostro(
    rostro,
    emocion=None
):

    try:

        if rostro is None:

            return None

        if not isinstance(
            rostro,
            np.ndarray
        ):

            return None

        if rostro.size == 0:

            return None

        # ==================================================
        # NOMBRE
        # ==================================================

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        milisegundos = int(
            (time.time() % 1) * 1000
        )

        if emocion:

            nombre_archivo = (
                f"rostro_"
                f"{timestamp}_"
                f"{milisegundos:03d}_"
                f"{emocion}.jpg"
            )

        else:

            nombre_archivo = (
                f"rostro_"
                f"{timestamp}_"
                f"{milisegundos:03d}.jpg"
            )

        ruta = os.path.join(
            ROSTROS_DIR,
            nombre_archivo
        )

        # ==================================================
        # GUARDAR
        # ==================================================

        guardado = cv2.imwrite(
            ruta,
            rostro
        )

        if guardado:

            logger.info(
                "📸 Rostro guardado: %s",
                ruta
            )

            return ruta

        logger.warning(
            "⚠️ No se pudo guardar el rostro."
        )

        return None

    except Exception as e:

        logger.warning(
            "⚠️ Error guardando rostro: %s",
            e
        )

        return None


# ==========================================================
# DIBUJAR RESULTADO
# ==========================================================

def dibujar_resultado(
    frame,
    x,
    y,
    w,
    h,
    emocion,
    confianza,
    todas
):

    # ======================================================
    # RECTÁNGULO
    # ======================================================

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # ======================================================
    # EMOCIÓN
    # ======================================================

    texto_principal = (
        f"{emocion}: "
        f"{confianza:.1f}%"
    )

    texto_y = max(
        30,
        y - 10
    )

    cv2.putText(
        frame,
        texto_principal,
        (x, texto_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ======================================================
    # PANEL
    # ======================================================

    panel_x = x + w + 10

    panel_y = y

    panel_width = 190

    panel_height = 130

    if (
        panel_x + panel_width
        >
        frame.shape[1]
    ):

        panel_x = (
            x
            -
            panel_width
            -
            10
        )

    if panel_x < 0:

        panel_x = 5

    if (
        panel_y + panel_height
        >
        frame.shape[0]
    ):

        panel_y = max(
            5,
            frame.shape[0]
            -
            panel_height
            -
            5
        )

    cv2.rectangle(
        frame,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (20, 20, 20),
        -1
    )

    # ======================================================
    # EMOCIONES
    # ======================================================

    for i, nombre in enumerate(
        EMOCIONES
    ):

        porcentaje = todas.get(
            nombre,
            0
        )

        texto = (
            f"{nombre}: "
            f"{porcentaje:.1f}%"
        )

        texto_y = (
            panel_y
            +
            25
            +
            i * 26
        )

        cv2.putText(
            frame,
            texto,
            (
                panel_x + 8,
                texto_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


# ==========================================================
# DIBUJAR CONSEJO
# ==========================================================

def dibujar_consejo(
    frame,
    consejo
):

    if consejo is None:

        return

    titulo = consejo.get(
        "titulo",
        "Consejo"
    )

    mensaje = consejo.get(
        "mensaje",
        ""
    )

    panel_x = 20
    panel_y = 60
    panel_width = 650
    panel_height = 110

    # ------------------------------------------------------
    # Ajustar al tamaño del frame
    # ------------------------------------------------------

    panel_width = min(
        panel_width,
        frame.shape[1] - 40
    )

    cv2.rectangle(
        frame,
        (
            panel_x,
            panel_y
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height
        ),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        f"Consejo: {titulo}",
        (
            panel_x + 15,
            panel_y + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # ======================================================
    # DIVIDIR MENSAJE
    # ======================================================

    palabras = mensaje.split()

    linea1 = ""

    linea2 = ""

    for palabra in palabras:

        if len(linea1) + len(palabra) < 70:

            linea1 += (
                palabra + " "
            )

        else:

            linea2 += (
                palabra + " "
            )

    # ======================================================
    # MOSTRAR LÍNEAS
    # ======================================================

    cv2.putText(
        frame,
        linea1.strip(),
        (
            panel_x + 15,
            panel_y + 60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    if linea2:

        cv2.putText(
            frame,
            linea2.strip(),
            (
                panel_x + 15,
                panel_y + 85
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


# ==========================================================
# PRUEBA LOCAL DEL MODELO
# ==========================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "🧠 EMOTISCAN - TFLITE"
    )

    print(
        "🔍 PRUEBA DE CARGA DEL MODELO"
    )

    print(
        "=" * 60
    )

    try:

        cargar_modelo()

        print()

        print(
            "✅ El modelo TFLite se cargó correctamente."
        )

        print(
            "📁 Modelo:",
            MODEL_PATH
        )

        print(
            "📥 Entrada:",
            input_details[0]["shape"]
        )

        print(
            "📤 Salida:",
            output_details[0]["shape"]
        )

        print(
            "🎭 Clases:",
            EMOCIONES
        )

        print(
            "=" * 60
        )

    except Exception as e:

        print()

        print(
            "❌ ERROR:"
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )