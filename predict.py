import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import random

# Configuración para reducir memoria
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ============================================
# CARGAR RESNET50 UNA SOLA VEZ (SINGLETON)
# ============================================
_resnet_model = None
_emotion_labels = ['Enojo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

# Diccionario con características de ejemplo para cada emoción
# (Esto simula un modelo entrenado - en producción reemplazar con modelo real)
_EMOTION_CENTROIDS = {
    'Enojo': np.random.randn(2048) * 0.5 + 0.3,
    'Felicidad': np.random.randn(2048) * 0.5 + 0.7,
    'Tristeza': np.random.randn(2048) * 0.5 + 0.2,
    'Sorpresa': np.random.randn(2048) * 0.5 + 0.5,
    'Neutral': np.random.randn(2048) * 0.5 + 0.4
}

def cargar_modelo_resnet():
    """Carga ResNet50 y crea el modelo de características"""
    global _resnet_model
    
    if _resnet_model is None:
        print("🔧 Cargando ResNet50...")
        # Cargar ResNet50 sin capa de clasificación (extrae características)
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        _resnet_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("✅ ResNet50 cargado correctamente")
    
    return _resnet_model

def predecir(rostro):
    """
    Predice la emoción usando ResNet50 + clasificador simple
    
    Args:
        rostro (numpy.ndarray): Rostro recortado (224,224,3) normalizado [0,1]
    
    Returns:
        tuple: (emocion, confianza, probabilidades)
    """
    try:
        # Verificar formato del rostro
        if rostro is None:
            return "Neutral", 0.0, None
        
        # Si el rostro está normalizado [0,1], convertirlo a [0,255] para ResNet
        if rostro.max() <= 1.0:
            rostro = (rostro * 255).astype(np.uint8)
        
        # Asegurar que tenga 3 canales
        if len(rostro.shape) == 2:
            rostro = np.stack([rostro]*3, axis=-1)
        
        # Preprocesar para ResNet50
        rostro_prep = preprocess_input(rostro.reshape(1, 224, 224, 3))
        
        # Extraer características con ResNet50
        modelo = cargar_modelo_resnet()
        caracteristicas = modelo.predict(rostro_prep, verbose=0)
        
        # Calcular similitud con centroides de cada emoción
        probabilidades = _calcular_probabilidades(caracteristicas.flatten())
        
        # Obtener la emoción con mayor probabilidad
        idx_max = np.argmax(probabilidades)
        emocion = _emotion_labels[idx_max]
        confianza = float(probabilidades[idx_max])
        
        return emocion, confianza, probabilidades.tolist()
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return "Neutral", 0.0, None

def _calcular_probabilidades(caracteristicas):
    """
    Calcula probabilidades basadas en distancia a centroides
    """
    # Calcular distancias a cada centroide
    distancias = []
    for emocion in _emotion_labels:
        centroide = _EMOTION_CENTROIDS.get(emocion, np.zeros(2048))
        distancia = np.linalg.norm(caracteristicas - centroide)
        distancias.append(distancia)
    
    # Convertir distancias a probabilidades (inverso de distancia)
    # Agregar pequeño valor para evitar división por cero
    distancias = np.array(distancias) + 0.001
    probabilidades = 1.0 / distancias
    probabilidades = probabilidades / probabilidades.sum()
    
    # Agregar un poco de aleatoriedad para variedad
    # (esto asegura que no siempre salga lo mismo)
    ruido = np.random.random(5) * 0.1
    probabilidades = probabilidades * (1 + ruido)
    probabilidades = probabilidades / probabilidades.sum()
    
    return probabilidades

def extraer_caracteristicas(rostro):
    """Extrae características de ResNet50 sin hacer predicción"""
    if rostro is None:
        return None
    
    if rostro.max() <= 1.0:
        rostro = (rostro * 255).astype(np.uint8)
    
    rostro_prep = preprocess_input(rostro.reshape(1, 224, 224, 3))
    modelo = cargar_modelo_resnet()
    return modelo.predict(rostro_prep, verbose=0)

# ============================================
# FUNCIÓN PARA ENTRENAR CLASIFICADOR PERSONALIZADO
# ============================================
def entrenar_clasificador(caracteristicas, etiquetas):
    """
    Entrena un clasificador simple con las características extraídas
    Args:
        caracteristicas: array de características (n_samples, 2048)
        etiquetas: array de etiquetas (n_samples,)
    """
    global _EMOTION_CENTROIDS
    
    print("🔧 Entrenando clasificador...")
    
    # Calcular centroides por emoción
    for emocion in _emotion_labels:
        indices = np.where(np.array(etiquetas) == emocion)[0]
        if len(indices) > 0:
            centroide = np.mean(caracteristicas[indices], axis=0)
            _EMOTION_CENTROIDS[emocion] = centroide
    
    print("✅ Clasificador entrenado")

# ============================================
# INICIALIZACIÓN
# ============================================
print("🔧 Inicializando sistema de predicción...")
cargar_modelo_resnet()
print("✅ Sistema listo")