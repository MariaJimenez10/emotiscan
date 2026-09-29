import cv2
import numpy as np
import tensorflow as tf

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_PATH = "modelo_resnet50_emociones.tflite"
EMOCIONES = ['Enojo', 'Felicidad', 'Neutral', 'Tristeza']

def predecir_emocion(ruta_imagen):
    """
    Función simple para predecir la emoción de una imagen
    """
    # 1. Cargar modelo
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # 2. Obtener detalles del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 3. Determinar formato esperado
    expected_shape = input_details[0]['shape']
    is_grayscale = expected_shape[-1] == 1
    target_size = (expected_shape[1], expected_shape[2])
    
    print(f"📥 Modelo espera: {expected_shape}")
    
    # 4. Cargar y preprocesar imagen
    face = cv2.imread(ruta_imagen)
    if face is None:
        print(f"❌ Error: No se pudo cargar la imagen {ruta_imagen}")
        return None
    
    # Preprocesar según formato
    if is_grayscale:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, target_size)
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=(0, -1))
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, target_size)
        face = face.astype(np.float32)
        
        # Si es ResNet50 (224x224), usar su preprocesamiento
        if target_size == (224, 224):
            from tensorflow.keras.applications.resnet50 import preprocess_input
            face = preprocess_input(face)
        else:
            face = face / 255.0
        
        face = np.expand_dims(face, axis=0)
    
    # 5. Realizar predicción
    interpreter.set_tensor(input_details[0]['index'], face)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # 6. Procesar salida (softmax si es necesario)
    if not np.isclose(np.sum(output), 1.0, atol=1e-3):
        exp_output = np.exp(output - np.max(output))
        output = exp_output / np.sum(exp_output)
    
    # 7. Mostrar resultados
    print("\n" + "=" * 50)
    print("🎭 RESULTADO")
    print("=" * 50)
    
    for i, emocion in enumerate(EMOCIONES):
        print(f"   {emocion}: {output[i]*100:.2f}%")
    
    principal = EMOCIONES[np.argmax(output)]
    confianza = np.max(output) * 100
    print(f"\n🏆 PRINCIPAL: {principal} ({confianza:.2f}%)")
    print("=" * 50)
    
    return output

# ============================================
# EJECUCIÓN (CAMBIAR RUTA DE IMAGEN)
# ============================================
if __name__ == "__main__":
    # Cambia esta ruta por la de tu imagen
    RUTA_IMAGEN = "tu_imagen.jpg"
    
    # Si no existe, crear imagen de prueba
    import os
    if not os.path.exists(RUTA_IMAGEN):
        print(f"⚠️ No se encuentra {RUTA_IMAGEN}, creando imagen de prueba...")
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite("test.jpg", test_img)
        RUTA_IMAGEN = "test.jpg"
    
    predecir_emocion(RUTA_IMAGEN)