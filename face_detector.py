import cv2
import numpy as np
from deepface import DeepFace

def detectar_rostro(img):
    """
    Detecta y recorta el rostro usando DeepFace
    Retorna: rostro recortado (224,224,3) o None
    """
    try:
        # Guardar imagen temporal para DeepFace
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, img)
        
        # DeepFace detecta y recorta automáticamente
        rostro = DeepFace.detectFace(
            img_path=temp_path,
            target_size=(224, 224),  # Tamaño para ResNet50
            detector_backend='opencv',  # Rápido y ligero
            enforce_detection=False  # No lanza error si no encuentra
        )
        
        # Eliminar archivo temporal
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return rostro
        
    except Exception as e:
        print(f"⚠️ Error detectando rostro: {e}")
        # Fallback: usar Haar Cascade
        return detectar_rostro_haar(img)

def detectar_rostro_haar(img):
    """
    Método alternativo usando Haar Cascade (fallback)
    """
    try:
        # Convertir a escala de grises
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detectar rostros con Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        caras = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(caras) == 0:
            print("⚠️ No se detectaron rostros")
            return None
        
        # Tomar el primer rostro detectado
        (x, y, w, h) = caras[0]
        rostro = img[y:y+h, x:x+w]
        
        # Redimensionar a 224x224
        rostro = cv2.resize(rostro, (224, 224))
        
        # Normalizar a [0,1] como espera DeepFace
        rostro = rostro / 255.0
        
        return rostro
        
    except Exception as e:
        print(f"❌ Error en detección alternativa: {e}")
        return None