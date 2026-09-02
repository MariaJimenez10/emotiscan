import cv2
import os
import logging

# ==========================================================
# CONFIGURACIÓN Y LOGGING
# ==========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_emociones.xml")

EMOCIONES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]
IMG_SIZE = (48, 48)

# ==========================================================
# VARIABLES GLOBALES (Carga Diferida)
# ==========================================================
emotion_recognizer = None
face_cascade = None

def cargar_modelos():
    """Carga los modelos en RAM solo cuando se necesite predecir."""
    global emotion_recognizer, face_cascade
    
    if emotion_recognizer is None:
        logger.info("📁 Cargando modelo de emociones desde %s", MODEL_PATH)

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ No se encontró el modelo en {MODEL_PATH}")

        emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
        emotion_recognizer.read(MODEL_PATH)
        logger.info("✅ Modelo LBPH cargado correctamente")

    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            raise RuntimeError("❌ No se pudo cargar Haar Cascade")
        logger.info("✅ Detector Haar Cascade cargado")

# ==========================================================
# FUNCIÓN PARA PREDECIR EMOCIÓN
# ==========================================================
def predecir(rostro):
    try:
        cargar_modelos()

        if rostro is None:
            return ("Neutral", 0.0, {"label": 2, "distancia": 0.0})

        # 1. Escala de grises en memoria
        if len(rostro.shape) == 3:
            rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        else:
            rostro_gray = rostro.copy()

        # 2. Redimensionar a 48x48
        rostro_gray = cv2.resize(rostro_gray, IMG_SIZE)

        # 3. Predicción con Modelo LBPH
        label, distancia = emotion_recognizer.predict(rostro_gray)

        if label < 0 or label >= len(EMOCIONES):
            return ("Neutral", 0.0, {"label": 2, "distancia": float(distancia)})

        emocion = EMOCIONES[label]
        confianza = max(0.0, min(100.0, 100.0 - distancia))

        return (
            emocion,
            confianza,
            {
                "label": int(label),
                "distancia": float(distancia)
            }
        )

    except Exception as e:
        logger.exception("❌ Error interno en predecir: %s", e)
        return ("Neutral", 0.0, {"error": str(e)})