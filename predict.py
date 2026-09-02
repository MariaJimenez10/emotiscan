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
DEBUG_DIR = os.path.join(BASE_DIR, "debug_rostros")

# Crear directorio de debug de forma segura
try:
    os.makedirs(DEBUG_DIR, exist_ok=True)
except Exception as e:
    logger.warning(f"No se pudo crear carpeta debug: {e}")

EMOCIONES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]
IMG_SIZE = (48, 48)

# ==========================================================
# VARIABLES GLOBALES (Carga Diferida / Lazy Loading)
# ==========================================================
emotion_recognizer = None
face_cascade = None

def cargar_modelos():
    """Carga los modelos en memoria solo cuando sea necesario para no colapsar la RAM al arrancar Gunicorn."""
    global emotion_recognizer, face_cascade
    
    if emotion_recognizer is None:
        logger.info("==========================================")
        logger.info("       CARGANDO MODELO DE EMOCIONES")
        logger.info("==========================================")
        logger.info(f"📁 Ubicación del modelo: {MODEL_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ No se encontró el modelo en {MODEL_PATH}")

        emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
        emotion_recognizer.read(MODEL_PATH)
        logger.info("✅ Modelo LBPH cargado correctamente en RAM")

    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            raise RuntimeError("❌ No se pudo cargar el detector Haar Cascade")
        logger.info("✅ Detector de rostros Haar Cascade cargado")

# ==========================================================
# FUNCIÓN PARA PREDECIR EMOCIÓN
# ==========================================================
def predecir(rostro):
    try:
        # Asegurar que los modelos estén cargados antes de ejecutar la predicción
        cargar_modelos()

        logger.info("🔍 INICIANDO PREDICCIÓN")

        if rostro is None:
            logger.error("❌ No se recibió ningún rostro")
            return ("Neutral", 0.0, None)

        logger.info(f"📷 Rostro recibido: {rostro.shape}, Tipo: {rostro.dtype}")

        # Guardar copia para debug si el entorno lo permite
        try:
            ruta_original = os.path.join(DEBUG_DIR, "01_rostro_recibido.jpg")
            cv2.imwrite(ruta_original, rostro)
        except Exception:
            pass

        # Convertir a escala de grises
        if len(rostro.shape) == 3:
            rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        else:
            rostro_gray = rostro.copy()

        # Redimensionar a 48x48 (mismo tamaño de entrenamiento)
        rostro_gray = cv2.resize(rostro_gray, IMG_SIZE)

        # Ejecutar Modelo LBPH
        label, distancia = emotion_recognizer.predict(rostro_gray)

        if label < 0 or label >= len(EMOCIONES):
            logger.error(f"❌ Label inválido: {label}")
            return ("Neutral", 0.0, None)

        emocion = EMOCIONES[label]
        confianza = max(0.0, min(100.0, 100.0 - distancia))

        logger.info(f"🏆 EMOCIÓN: {emocion} | DISTANCIA: {distancia:.4f} | CONFIANZA: {confianza:.2f}%")

        return (
            emocion,
            confianza,
            {
                "label": int(label),
                "distancia": float(distancia)
            }
        )

    except Exception as e:
        logger.exception(f"❌ Error en predicción: {e}")
        return ("Neutral", 0.0, None)

# ==========================================================
# EJECUTAR CÁMARA (MODO LOCAL)
# ==========================================================
if __name__ == "__main__":
    logger.info("🎥 INICIANDO CÁMARA EN MODO LOCAL")
    cargar_modelos()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in rostros:
            rostro = frame[y:y+h, x:x+w]
            emocion, confianza, datos = predecir(rostro)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emocion} {confianza:.1f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("EmotiScan - Reconocimiento de emociones", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()