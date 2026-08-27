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
# RUTAS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG_DIR = os.path.join(
    BASE_DIR,
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
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)


logger.info("======================================")
logger.info("🧠 CARGANDO MODELO TFLITE")
logger.info(f"📁 Ubicación del modelo: {MODEL_PATH}")


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"❌ No existe el modelo: {MODEL_PATH}"
    )


# ==========================================================
# CARGAR INTERPRETER
# ==========================================================

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()


# ==========================================================
# INFORMACIÓN DEL MODELO
# ==========================================================

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


logger.info("✅ Modelo TFLite cargado correctamente")

logger.info(
    f"📥 Entrada modelo: {input_details[0]['shape']}"
)

logger.info(
    f"📥 Tipo entrada: {input_details[0]['dtype']}"
)

logger.info(
    f"📤 Salida modelo: {output_details[0]['shape']}"
)

logger.info(
    f"📤 Tipo salida: {output_details[0]['dtype']}"
)

logger.info("======================================")


# ==========================================================
# EMOCIONES
# IMPORTANTE:
# ESTE ORDEN DEBE SER EXACTAMENTE EL MISMO
# QUE SE UTILIZÓ DURANTE EL ENTRENAMIENTO
# ==========================================================

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# PREDECIR EMOCIÓN
# ==========================================================

def predecir(rostro):

    try:

        logger.info("")
        logger.info("======================================")
        logger.info("🔍 INICIANDO PREDICCIÓN")
        logger.info("======================================")


        # ==================================================
        # VERIFICAR QUE EL ROSTRO EXISTA
        # ==================================================

        if rostro is None:

            logger.error(
                "❌ ERROR: No se recibió ningún rostro"
            )

            return (
                "Neutral",
                0.0,
                None
            )


        # ==================================================
        # INFORMACIÓN DEL ROSTRO RECIBIDO
        # ==================================================

        logger.info(
            f"📷 Rostro recibido: {rostro.shape}"
        )

        logger.info(
            f"📷 Tipo: {rostro.dtype}"
        )

        logger.info(
            f"📷 MIN: {rostro.min()}"
        )

        logger.info(
            f"📷 MAX: {rostro.max()}"
        )


        # ==================================================
        # GUARDAR ROSTRO ORIGINAL RECIBIDO
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
            f"📸 Rostro original guardado: "
            f"{ruta_original}"
        )


        # ==================================================
        # REDIMENSIONAR
        # ==================================================

        rostro = cv2.resize(
            rostro,
            (224, 224)
        )

        logger.info(
            f"📐 Después de resize: {rostro.shape}"
        )


        # ==================================================
        # GUARDAR ROSTRO REDIMENSIONADO
        # TODAVÍA EN FORMATO BGR
        # ==================================================

        ruta_resize = os.path.join(
            DEBUG_DIR,
            "02_rostro_resize_bgr.jpg"
        )

        cv2.imwrite(
            ruta_resize,
            rostro
        )


        # ==================================================
        # CONVERTIR BGR -> RGB
        # IMPORTANTE PARA RESNET50
        # ==================================================

        rostro = cv2.cvtColor(
            rostro,
            cv2.COLOR_BGR2RGB
        )

        logger.info(
            "🎨 Conversión BGR -> RGB realizada"
        )


        # ==================================================
        # GUARDAR ROSTRO RGB
        #
        # OpenCV guarda esperando BGR, por eso convertimos
        # nuevamente solo para poder visualizarlo correctamente
        # ==================================================

        ruta_rgb = os.path.join(
            DEBUG_DIR,
            "03_rostro_rgb.jpg"
        )

        rostro_visual = cv2.cvtColor(
            rostro,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            ruta_rgb,
            rostro_visual
        )

        del rostro_visual


        # ==================================================
        # CONVERTIR A FLOAT32
        # ==================================================

        rostro = rostro.astype(
            np.float32
        )

        logger.info(
            f"🔢 Después de float32: {rostro.dtype}"
        )


        # ==================================================
        # PREPROCESAMIENTO RESNET50
        # ==================================================

        rostro = preprocess_input(
            rostro
        )

        logger.info(
            "🧠 Preprocesamiento ResNet50 realizado"
        )

        logger.info(
            f"📊 MIN después preprocess: "
            f"{rostro.min():.2f}"
        )

        logger.info(
            f"📊 MAX después preprocess: "
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
            f"➡️ Entrada FINAL al modelo: "
            f"{rostro.shape}"
        )


        # ==================================================
        # VERIFICAR FORMA ESPERADA POR EL MODELO
        # ==================================================

        forma_modelo = tuple(
            input_details[0]["shape"]
        )

        forma_rostro = tuple(
            rostro.shape
        )


        logger.info(
            f"🧠 Modelo espera: {forma_modelo}"
        )

        logger.info(
            f"➡️ Estamos enviando: {forma_rostro}"
        )


        if forma_modelo != forma_rostro:

            logger.error(
                "❌ ERROR: Las dimensiones NO coinciden"
            )

            return (
                "Neutral",
                0.0,
                None
            )


        # ==================================================
        # VERIFICAR TIPO DE DATO ESPERADO
        # ==================================================

        tipo_esperado = input_details[0]["dtype"]

        if rostro.dtype != tipo_esperado:

            logger.warning(
                f"⚠️ Convirtiendo entrada de "
                f"{rostro.dtype} a {tipo_esperado}"
            )

            rostro = rostro.astype(
                tipo_esperado
            )


        # ==================================================
        # ENVIAR IMAGEN AL MODELO
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

        logger.info(
            "🚀 EJECUTANDO MODELO..."
        )

        interpreter.invoke()

        logger.info(
            "✅ Modelo ejecutado correctamente"
        )


        # ==================================================
        # OBTENER PREDICCIÓN
        # ==================================================

        pred = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]


        logger.info(
            f"🔥 PREDICCIÓN CRUDA: {pred}"
        )


        # ==================================================
        # VERIFICAR NÚMERO DE CLASES
        # ==================================================

        if len(pred) != len(EMOCIONES):

            logger.error(
                f"❌ El modelo devuelve {len(pred)} "
                f"clases, pero tenemos "
                f"{len(EMOCIONES)} emociones"
            )

            return (
                "Neutral",
                0.0,
                pred.tolist()
            )


        # ==================================================
        # MOSTRAR PROBABILIDADES DE TODAS LAS EMOCIONES
        # ==================================================

        logger.info("")
        logger.info("======================================")
        logger.info("📊 PROBABILIDADES POR EMOCIÓN")
        logger.info("======================================")


        for i, valor in enumerate(pred):

            porcentaje = float(valor) * 100

            logger.info(
                f"{i} - {EMOCIONES[i]}: "
                f"{float(valor):.6f} "
                f"({porcentaje:.2f}%)"
            )


        logger.info("======================================")


        # ==================================================
        # OBTENER ÍNDICE GANADOR
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
        # OBTENER CONFIANZA
        # ==================================================

        confianza = float(
            pred[indice]
        )


        logger.info(
            f"🏆 ÍNDICE GANADOR: {indice}"
        )

        logger.info(
            f"🏆 EMOCIÓN FINAL: {emocion}"
        )

        logger.info(
            f"🏆 CONFIANZA: "
            f"{confianza:.6f} "
            f"({confianza * 100:.2f}%)"
        )


        # ==================================================
        # GUARDAR RESULTADO VISUAL
        # ==================================================

        ruta_resultado = os.path.join(
            DEBUG_DIR,
            "05_resultado_emocion.jpg"
        )


        imagen_resultado = cv2.imread(
            ruta_resize
        )


        if imagen_resultado is not None:

            texto = (
                f"{emocion} "
                f"({confianza * 100:.1f}%)"
            )


            cv2.putText(
                imagen_resultado,
                texto,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            cv2.imwrite(
                ruta_resultado,
                imagen_resultado
            )


            logger.info(
                f"📸 Resultado guardado: "
                f"{ruta_resultado}"
            )


        # ==================================================
        # FINALIZAR
        # ==================================================

        logger.info(
            "======================================"
        )

        logger.info(
            "✅ PREDICCIÓN FINALIZADA"
        )

        logger.info(
            "======================================"
        )


        resultado_pred = pred.tolist()


        # ==================================================
        # LIMPIAR MEMORIA
        # ==================================================

        del rostro
        gc.collect()


        # ==================================================
        # RETORNAR RESULTADO
        # ==================================================

        return (
            emocion,
            confianza,
            resultado_pred
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