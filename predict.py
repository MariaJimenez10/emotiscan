import numpy as np
import cv2
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predecir(rostro):
    """Clasifica emoción basado en colores y formas (sin IA)"""
    try:
        if rostro is None:
            return "Neutral", 0.3, None
        
        # Convertir a uint8
        if rostro.max() <= 1.0:
            rostro = (rostro * 255).astype(np.uint8)
        
        # Extraer características
        emocion, confianza = _analizar_rostro(rostro)
        
        # Crear probabilidades simuladas
        labels = ['Enojo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']
        probas = [0.0] * 5
        idx = labels.index(emocion) if emocion in labels else 4
        probas[idx] = confianza
        
        # Distribuir el resto
        resto = 1.0 - confianza
        for i in range(5):
            if i != idx:
                probas[i] = resto / 4
        
        return emocion, confianza, probas
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return "Neutral", 0.3, None

def _analizar_rostro(rostro):
    """Análisis simple por colores y textura"""
    h, w, _ = rostro.shape
    
    # Convertir a HSV
    hsv = cv2.cvtColor(rostro, cv2.COLOR_RGB2HSV)
    
    # Estadísticas generales
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # Zona de la boca (inferior)
    boca = rostro[int(h*0.6):int(h*0.85), int(w*0.2):int(w*0.8)]
    if boca.size > 0:
        boca_v = np.mean(boca[:, :, 2]) if len(boca.shape) == 3 else np.mean(boca)
    else:
        boca_v = 0.5
    
    # Zona de los ojos (superior)
    ojos = rostro[int(h*0.1):int(h*0.35), int(w*0.1):int(w*0.9)]
    if ojos.size > 0:
        ojos_v = np.mean(ojos[:, :, 2]) if len(ojos.shape) == 3 else np.mean(ojos)
    else:
        ojos_v = 0.5
    
    # Contraste
    contraste = np.std(rostro)
    
    # Clasificación
    scores = {
        'Felicidad': 0.1 + (boca_v if boca_v > 0.5 else 0) * 0.5,
        'Tristeza': 0.1 + (1 - boca_v if boca_v < 0.5 else 0) * 0.5,
        'Enojo': 0.1 + (contraste / 255) * 0.6,
        'Sorpresa': 0.1 + (ojos_v if ojos_v > 0.6 else 0) * 0.5,
        'Neutral': 0.2 + (1 - abs(v_mean - 0.5)) * 0.3
    }
    
    # Agregar ruido para variedad
    for key in scores:
        scores[key] += random.random() * 0.15
    
    # Normalizar
    total = sum(scores.values())
    if total > 0:
        for key in scores:
            scores[key] = scores[key] / total
    
    # Obtener mejor resultado
    emocion = max(scores, key=scores.get)
    confianza = scores[emocion]
    
    # Asegurar confianza mínima
    confianza = max(0.2, min(0.9, confianza))
    
    logger.info(f"🎯 {emocion} ({confianza:.2f})")
    return emocion, confianza