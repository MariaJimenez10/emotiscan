import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detectar_rostro(img):
    """Detecta rostro usando Haar Cascade (muy ligero)"""
    try:
        if img is None:
            return None
        
        # Convertir a grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Cargar clasificador
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detectar
        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)  # Reducido para mejor rendimiento
        )
        
        if len(caras) == 0:
            logger.warning("⚠️ No se detectaron rostros")
            return None
        
        # Tomar el primer rostro
        (x, y, w, h) = caras[0]
        rostro = img[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (224, 224))
        rostro = rostro.astype(np.float32) / 255.0
        
        logger.info(f"✅ Rostro detectado: {w}x{h}")
        return rostro
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None