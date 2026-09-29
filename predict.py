import os
import logging
import cv2
import numpy as np
import tensorflow as tf
import time
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


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
# MODELO
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo",
    "mejor_modelo.keras"
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
#
# IMPORTANTE:
#
# Este orden debe ser EXACTAMENTE el mismo
# que utilizó el generador durante el entrenamiento.
#
# En tu dataset:
#
# Enojo
# Felicidad
# Neutral
# Tristeza
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
# VARIABLE GLOBAL DEL MODELO
# ==========================================================

model = None


# ==========================================================
# VARIABLES PARA EL CONSEJO
# ==========================================================

ultima_emocion_consejo = None
consejo_actual = None


# ==========================================================
# CARGAR MODELO
# ==========================================================

def cargar_modelo():

    global model

    logger.info("=" * 60)

    logger.info(
        "🧠 CARGANDO MODELO RESNET50"
    )

    logger.info("=" * 60)

    logger.info(
        f"📁 Modelo: {MODEL_PATH}"
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No existe el modelo:\n{MODEL_PATH}"
        )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    logger.info(
        "✅ MODELO CARGADO CORRECTAMENTE"
    )

    logger.info(
        f"📥 Entrada esperada: {model.input_shape}"
    )

    logger.info(
        f"📤 Salida: {model.output_shape}"
    )

    logger.info(
        f"🎭 Orden de clases utilizado:"
    )

    for i, emocion in enumerate(EMOCIONES):

        logger.info(
            f"   Índice {i} → {emocion}"
        )

    logger.info(
        f"📂 Rostros guardados en: {ROSTROS_DIR}"
    )

    logger.info("=" * 60)

    return model


# ==========================================================
# OBTENER CONSEJO
# ==========================================================

def obtener_consejo(emocion):

    global ultima_emocion_consejo
    global consejo_actual

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
# PREPARAR IMAGEN
# ==========================================================

def preparar_imagen(rostro):

    if rostro is None:

        raise ValueError(
            "El rostro recibido es None."
        )

    if rostro.size == 0:

        raise ValueError(
            "El rostro está vacío."
        )

    # ======================================================
    # DETERMINAR CANALES
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

        # OpenCV recibe BGR.
        # ResNet50 espera RGB.

        rostro_rgb = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

    else:

        raise ValueError(
            f"Formato de imagen no soportado: "
            f"{rostro.shape}"
        )

    # ======================================================
    # REDIMENSIONAR
    # ======================================================

    rostro_rgb = cv2.resize(
        rostro_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # ======================================================
    # FLOAT32
    # ======================================================

    rostro_rgb = rostro_rgb.astype(
        np.float32
    )

    # ======================================================
    # PREPROCESAMIENTO RESNET50
    # ======================================================

    rostro_rgb = preprocess_input(
        rostro_rgb
    )

    # ======================================================
    # AGREGAR BATCH
    # ======================================================

    rostro_rgb = np.expand_dims(
        rostro_rgb,
        axis=0
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

    exp_x = np.exp(x)

    return (
        exp_x /
        np.sum(exp_x)
    )


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(rostro):

    global model

    if model is None:

        cargar_modelo()

    # ======================================================
    # PREPARAR IMAGEN
    # ======================================================

    imagen = preparar_imagen(
        rostro
    )

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    predicciones = model.predict(
        imagen,
        verbose=0
    )

    probabilidades = np.squeeze(
        predicciones
    )

    probabilidades = np.asarray(
        probabilidades,
        dtype=np.float32
    )

    # ======================================================
    # COMPROBAR SALIDA
    # ======================================================

    if len(probabilidades) != len(
        EMOCIONES
    ):

        raise ValueError(
            f"El modelo devuelve "
            f"{len(probabilidades)} clases, "
            f"pero EMOCIONES tiene "
            f"{len(EMOCIONES)}."
        )

    # ======================================================
    # CONVERTIR A PROBABILIDADES
    # ======================================================

    suma = float(
        np.sum(probabilidades)
    )

    if (
        np.all(probabilidades >= 0)
        and
        np.all(probabilidades <= 1)
        and
        abs(suma - 1.0) < 0.05
    ):

        if suma > 0:

            probabilidades = (
                probabilidades / suma
            )

    else:

        probabilidades = softmax(
            probabilidades
        )

    # ======================================================
    # MOSTRAR PROBABILIDADES REALES
    # ======================================================

    logger.info("")
    logger.info("==========================================")
    logger.info("📊 PREDICCIÓN DEL MODELO")
    logger.info("==========================================")

    for i, nombre in enumerate(EMOCIONES):

        logger.info(
            f"{nombre:<12}: "
            f"{probabilidades[i] * 100:6.2f}%"
        )

    # ======================================================
    # EMOCIÓN PRINCIPAL
    # ======================================================

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

    logger.info("------------------------------------------")

    logger.info(
        f"🎯 EMOCIÓN: {emocion}"
    )

    logger.info(
        f"📈 CONFIANZA: {confianza:.2f}%"
    )

    logger.info(
        f"🔢 ÍNDICE: {indice}"
    )

    logger.info("==========================================")

    # ======================================================
    # TODAS LAS EMOCIONES
    # ======================================================

    todas = {}

    for i, nombre in enumerate(
        EMOCIONES
    ):

        todas[nombre] = round(
            float(
                probabilidades[i] * 100
            ),
            2
        )

    # ======================================================
    # CONSEJO
    # ======================================================

    consejo = obtener_consejo(
        emocion
    )

    # ======================================================
    # RESULTADO
    # ======================================================

    return (
        emocion,
        confianza,
        todas,
        consejo
    )


# ==========================================================
# GUARDAR ROSTRO
# ==========================================================

def guardar_rostro(
    rostro,
    emocion=None
):

    if rostro is None:

        return None

    if rostro.size == 0:

        return None

    # ======================================================
    # GENERAR NOMBRE
    # ======================================================

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

    # ======================================================
    # GUARDAR
    # ======================================================

    guardado = cv2.imwrite(
        ruta,
        rostro
    )

    if guardado:

        logger.info(
            f"📸 Rostro guardado: {ruta}"
        )

        return ruta

    logger.warning(
        "⚠️ No se pudo guardar el rostro."
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
    # EMOCIÓN PRINCIPAL
    # ======================================================

    texto_principal = (
        f"{emocion}: "
        f"{confianza:.1f}%"
    )

    cv2.rectangle(
        frame,
        (x, y - 40),
        (x + w, y),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        texto_principal,
        (x + 5, y - 12),
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
# CÁMARA EN VIVO
# ==========================================================

def iniciar_camara():

    logger.info("")
    logger.info("=" * 60)
    logger.info("📷 INICIANDO CÁMARA")
    logger.info("=" * 60)

    logger.info(
        "🎥 Presiona Q para salir."
    )

    logger.info(
        "🙂 Coloca tu rostro frente a la cámara."
    )

    logger.info(
        "💡 Los consejos aparecerán según la emoción."
    )

    logger.info(
        f"📂 Los rostros se guardarán en:"
    )

    logger.info(
        f"   {ROSTROS_DIR}"
    )

    logger.info("=" * 60)

    # ======================================================
    # DETECTOR
    # ======================================================

    cascade_path = (
        cv2.data.haarcascades
        +
        "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    if face_detector.empty():

        raise RuntimeError(
            "No se pudo cargar el detector de rostros."
        )

    # ======================================================
    # CÁMARA
    # ======================================================

    camera = cv2.VideoCapture(
        0
    )

    if not camera.isOpened():

        raise RuntimeError(
            "❌ No se pudo abrir la cámara."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # ======================================================
    # CONTROL
    # ======================================================

    contador = 0

    resultado_anterior = None

    ultimo_guardado = 0

    PREDICT_EVERY = 3

    try:

        while True:

            # ==================================================
            # LEER FRAME
            # ==================================================

            ret, frame = camera.read()

            if not ret:

                logger.error(
                    "❌ No se pudo leer la cámara."
                )

                break

            # ==================================================
            # ESPEJO
            # ==================================================

            frame = cv2.flip(
                frame,
                1
            )

            # ==================================================
            # GRISES
            # ==================================================

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # ==================================================
            # DETECTAR ROSTROS
            # ==================================================

            rostros = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            # ==================================================
            # PROCESAR ROSTROS
            # ==================================================

            for (
                x,
                y,
                w,
                h
            ) in rostros:

                # ------------------------------------------
                # MARGEN
                # ------------------------------------------

                margen = int(
                    min(w, h) * 0.15
                )

                x1 = max(
                    0,
                    x - margen
                )

                y1 = max(
                    0,
                    y - margen
                )

                x2 = min(
                    frame.shape[1],
                    x + w + margen
                )

                y2 = min(
                    frame.shape[0],
                    y + h + margen
                )

                rostro = frame[
                    y1:y2,
                    x1:x2
                ]

                # ------------------------------------------
                # PREDICCIÓN
                # ------------------------------------------

                if (
                    contador % PREDICT_EVERY == 0
                    or resultado_anterior is None
                ):

                    try:

                        resultado_anterior = predecir(
                            rostro
                        )

                    except Exception as e:

                        logger.error(
                            f"❌ Error prediciendo: {e}"
                        )

                        resultado_anterior = None

                # ------------------------------------------
                # GUARDAR ROSTRO
                # ------------------------------------------

                tiempo_actual = time.time()

                if (
                    tiempo_actual
                    -
                    ultimo_guardado
                    >=
                    INTERVALO_GUARDADO
                ):

                    emocion_guardado = None

                    if resultado_anterior is not None:

                        emocion_guardado = (
                            resultado_anterior[0]
                        )

                    guardar_rostro(
                        rostro,
                        emocion_guardado
                    )

                    ultimo_guardado = tiempo_actual

                # ------------------------------------------
                # MOSTRAR RESULTADO
                # ------------------------------------------

                if resultado_anterior is not None:

                    (
                        emocion,
                        confianza,
                        todas,
                        consejo
                    ) = resultado_anterior

                    dibujar_resultado(
                        frame,
                        x,
                        y,
                        w,
                        h,
                        emocion,
                        confianza,
                        todas
                    )

                    dibujar_consejo(
                        frame,
                        consejo
                    )

                else:

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Analizando...",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # ==================================================
            # SIN ROSTRO
            # ==================================================

            if len(rostros) == 0:

                resultado_anterior = None

                cv2.putText(
                    frame,
                    "No se detecta rostro",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # ==================================================
            # INFORMACIÓN
            # ==================================================

            cv2.putText(
                frame,
                "Presiona Q para salir",
                (
                    20,
                    frame.shape[0] - 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ==================================================
            # MOSTRAR
            # ==================================================

            cv2.imshow(
                "EmotiScan - ResNet50",
                frame
            )

            # ==================================================
            # TECLADO
            # ==================================================

            tecla = (
                cv2.waitKey(1)
                &
                0xFF
            )

            if tecla == ord("q"):

                break

            contador += 1

    finally:

        camera.release()

        cv2.destroyAllWindows()

        logger.info("")
        logger.info(
            "📷 Cámara cerrada."
        )


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "🧠 EMOTISCAN - RESNET50"
    )

    print(
        "🎥 RECONOCIMIENTO DE EMOCIONES EN VIVO"
    )

    print(
        "📸 GUARDADO AUTOMÁTICO DE ROSTROS"
    )

    print(
        "💡 CONSEJOS PARA LAS 4 EMOCIONES"
    )

    print("=" * 60)

    try:

        cargar_modelo()

        iniciar_camara()

    except KeyboardInterrupt:

        print("")

        print(
            "⛔ Programa detenido."
        )

    except Exception as e:

        print("")

        print("=" * 60)

        print("❌ ERROR")

        print("=" * 60)

        print(e)

        print("=" * 60)
