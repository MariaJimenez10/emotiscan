# face_detector.py - VERSIÓN SIN DEEPFACE
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detectar_rostro(img):
    """
    Detecta rostro usando solo OpenCV (sin DeepFace)
    """
    if img is None:
        return None
    
    try:
        # Convertir a grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detector Haar
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detectar múltiples escalas
        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(caras) == 0:
            # Intentar con parámetros más flexibles
            caras = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
        
        if len(caras) == 0:
            logger.warning("⚠️ No se detectaron rostros")
            return None
        
        # Tomar el rostro más grande
        if len(caras) > 1:
            areas = [w * h for (x, y, w, h) in caras]
            idx = np.argmax(areas)
            (x, y, w, h) = caras[idx]
        else:
            (x, y, w, h) = caras[0]
        
        # Recortar
        rostro = img[y:y+h, x:x+w]
        
        # Redimensionar
        rostro = cv2.resize(rostro, (224, 224))
        
        # Normalizar
        rostro = rostro.astype(np.float32) / 255.0
        
        return rostro
        
    except Exception as e:
        logger.error(f"❌ Error detectando rostro: {e}")
        return None