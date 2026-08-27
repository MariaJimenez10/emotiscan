import os
import cv2
import gc
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ==========================================================
# CARPETA DEBUG
# ==========================================================

DEBUG_DIR = os.path.join(
    os.path.dirname(__file__),
    "debug_rostros"
)

os.makedirs(
    DEBUG_DIR,
    exist_ok=True
)


# ==========================================================
# MODELO TFLITE
# ==========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelo_resnet50_emociones.tflite"
)


logger.info(
    "======================================"
)

logger.info(
    "🧠 CARGANDO MODELO TFLITE"
)

logger.info(
    f"Modelo: {MODEL_PATH}"
)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"No existe el modelo: {MODEL_PATH}"
    )


interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()


# ==========================================================
# INFORMACIÓN DEL MODELO
# ==========================================================

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()


logger.info(
    "✅ Modelo TFLite cargado correctamente"
)


logger.info(
    f"Entrada modelo: {input_details[0]['shape']}"
)

logger.info(
    f"Tipo entrada: {input_details[0]['dtype']}"
)

logger.info(
    f"Salida modelo: {output_details[0]['shape']}"
)

logger.info(
    f"Tipo salida: {output_details[0]['dtype']}"
)

logger.info(
    "======================================"
)


# ==========================================================
# EMOCIONES
# ==========================================================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# PREDECIR
# ==========================================================

def predecir(rostro):

    try:

        logger.info(
            "======================================"
        )

        logger.info(
            "🔍 INICIANDO PREDICCIÓN"
        )


        # ==================================================
        # INFORMACIÓN DEL ROSTRO RECIBIDO
        # ==================================================

        logger.info(
            f"Rostro recibido: {rostro.shape}"
        )

        logger.info(
            f"Tipo: {rostro.dtype}"
        )

        logger.info(
            f"MIN: {rostro.min()}"
        )

        logger.info(
            f"MAX: {rostro.max()}"
        )


        # ==================================================
        # REDIMENSIONAR
        # ==================================================

        rostro = cv2.resize(
            rostro,
            (224, 224)
        )


        logger.info(
            f"Después de resize: {rostro.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO ANTES DEL PREPROCESAMIENTO
        # ==================================================

        ruta_debug = os.path.join(
            DEBUG_DIR,
            "04_rostro_entrada_visual.jpg"
        )

        cv2.imwrite(
            ruta_debug,
            rostro
        )


        # ==================================================
        # CONVERTIR FLOAT32
        # ==================================================

        rostro = rostro.astype(
            np.float32
        )


        logger.info(
            f"Después de float32: {rostro.dtype}"
        )


        # ==================================================
        # PREPROCESS RESNET50
        # ==================================================

        rostro = preprocess_input(
            rostro
        )


        logger.info(
            "Preprocesamiento ResNet50 realizado"
        )

        logger.info(
            f"MIN después preprocess: "
            f"{rostro.min():.2f}"
        )

        logger.info(
            f"MAX después preprocess: "
            f"{rostro.max():.2f}"
        )


        # ==================================================
        # AGREGAR DIMENSIÓN BATCH
        # ==================================================

        rostro = np.expand_dims(
            rostro,
            axis=0
        )


        logger.info(
            f"Entrada FINAL al modelo: "
            f"{rostro.shape}"
        )


        # ==================================================
        # VERIFICAR FORMA ESPERADA
        # ==================================================

        forma_modelo = tuple(
            input_details[0]["shape"]
        )

        forma_rostro = tuple(
            rostro.shape
        )


        logger.info(
            f"Modelo espera: {forma_modelo}"
        )

        logger.info(
            f"Estamos enviando: {forma_rostro}"
        )


        if forma_modelo != forma_rostro:

            logger.error(
                "❌ ERROR: Las dimensiones "
                "NO coinciden."
            )

            return (
                "Neutral",
                0.0,
                None
            )


        # ==================================================
        # ENVIAR AL MODELO
        # ==================================================

        interpreter.set_tensor(
            input_details[0]["index"],
            rostro
        )


        logger.info(
            "➡️ Imagen enviada al modelo"
        )


        # ==================================================
        # EJECUTAR MODELO
        # ==================================================

        logger.info("🚀 ANTES DE EJECUTAR MODELO")

        interpreter.invoke()

        logger.info("✅ DESPUÉS DE EJECUTAR MODELO")

        pred = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

        logger.info(f"🔥 PREDICCIÓN CRUDA: {pred}")



        logger.info(
            "✅ Modelo ejecutado"
        )


        # ==================================================
        # OBTENER PREDICCIÓN
        # ==================================================

        pred = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]


        logger.info(
            f"Predicción cruda: {pred}"
        )


        # ==================================================
        # VERIFICAR NÚMERO DE CLASES
        # ==================================================

        if len(pred) != len(EMOCIONES):

            logger.error(
                f"❌ El modelo devuelve "
                f"{len(pred)} clases, pero "
                f"tenemos {len(EMOCIONES)} emociones."
            )

            return (
                "Neutral",
                0.0,
                pred.tolist()
            )


        # ==================================================
        # OBTENER ÍNDICE
        # ==================================================

        indice = int(
            np.argmax(pred)
        )


        # ==================================================
        # OBTENER EMOCIÓN
        # ==================================================

        emocion = EMOCIONES[
            indice
        ]


        # ==================================================
        # CONFIANZA
        # ==================================================

        confianza = float(
            pred[indice]
        )


        logger.info(
            f"🎯 Índice: {indice}"
        )

        logger.info(
            f"🎯 Emoción: {emocion}"
        )

        logger.info(
            f"🎯 Confianza: {confianza:.4f}"
        )


        # ==================================================
        # GUARDAR RESULTADO VISUAL
        # ==================================================

        imagen_resultado = cv2.imread(
            ruta_debug
        )


        if imagen_resultado is not None:

            texto = (
                f"{emocion} "
                f"({confianza:.2f})"
            )


            cv2.putText(
                imagen_resultado,
                texto,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            ruta_resultado = os.path.join(
                DEBUG_DIR,
                "05_resultado_emocion.jpg"
            )


            cv2.imwrite(
                ruta_resultado,
                imagen_resultado
            )


            logger.info(
                f"📸 Resultado guardado: "
                f"{ruta_resultado}"
            )


        logger.info(
            "======================================"
        )


        # ==================================================
        # LIMPIAR
        # ==================================================

        del rostro

        gc.collect()


        return (
            emocion,
            confianza,
            pred.tolist()
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