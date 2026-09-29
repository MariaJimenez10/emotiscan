import cv2
import numpy as np
import tensorflow as tf
import os

# ============================================
# CONFIGURACIÓN
# ============================================
MODEL_PATH = "modelo_resnet50_emociones.tflite"
EMOCIONES = ['Enojo', 'Felicidad', 'Neutral', 'Tristeza']

def predecir_emocion(ruta_imagen):
    """
    Función CORREGIDA para predecir emociones con ResNet50
    """
    print("=" * 60)
    print("🎭 PREDICTOR DE EMOCIONES - RESNET50")
    print("=" * 60)
    
    # 1. Cargar modelo TFLite
    print("\n🧠 Cargando modelo...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ Modelo cargado")
    print(f"📥 Espera: {input_details[0]['shape']}")
    print(f"📤 Salida: {output_details[0]['shape']}")
    
    # 2. Cargar imagen
    print(f"\n📷 Cargando imagen: {ruta_imagen}")
    face = cv2.imread(ruta_imagen)
    
    if face is None:
        print(f"❌ Error: No se pudo cargar la imagen")
        return None
    
    print(f"✅ Imagen cargada: {face.shape}")
    
    # 3. PREPROCESAMIENTO CORRECTO para ResNet50
    print("\n🔄 Preprocesando imagen...")
    
    # Convertir BGR (OpenCV) a RGB
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a 224x224 (EXACTO como espera el modelo)
    face_resized = cv2.resize(face_rgb, (224, 224))
    
    # Convertir a float32
    face_float = face_resized.astype(np.float32)
    
    # APLICAR PREPROCESS_INPUT DE RESNET50 (¡IMPORTANTE!)
    # Esto normaliza correctamente los valores
    from tensorflow.keras.applications.resnet50 import preprocess_input
    face_preprocessed = preprocess_input(face_float)
    
    # Añadir dimensión de batch
    face_batch = np.expand_dims(face_preprocessed, axis=0)
    
    print(f"✅ Imagen procesada: {face_batch.shape}")
    print(f"📊 Rango de valores: [{face_batch.min():.2f}, {face_batch.max():.2f}]")
    
    # 4. Realizar predicción
    print("\n⏳ Prediciendo...")
    interpreter.set_tensor(input_details[0]['index'], face_batch)
    interpreter.invoke()
    
    # Obtener salida
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Aplicar softmax para obtener probabilidades
    probabilities = output[0]
    # Si no suman 1, aplicar softmax
    if not np.isclose(np.sum(probabilities), 1.0, atol=1e-3):
        exp_output = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_output / np.sum(exp_output)
    
    # 5. Mostrar resultados
    print("\n" + "=" * 60)
    print("🎭 RESULTADO")
    print("=" * 60)
    
    resultados = {}
    for i, emocion in enumerate(EMOCIONES):
        porcentaje = probabilities[i] * 100
        resultados[emocion] = porcentaje
        print(f"   {emocion}: {porcentaje:.2f}%")
    
    # Emoción principal
    idx_principal = np.argmax(probabilities)
    emocion_principal = EMOCIONES[idx_principal]
    confianza = probabilities[idx_principal] * 100
    
    print("\n" + "=" * 60)
    print(f"🏆 EMOCIÓN PRINCIPAL: {emocion_principal} ({confianza:.2f}%)")
    print("=" * 60)
    
    return {
        'emocion': emocion_principal,
        'confianza': confianza,
        'todas': resultados
    }

# ============================================
# FUNCIÓN PARA DETECTAR ROSTRO (OPCIONAL)
# ============================================
def detectar_y_predecir(ruta_imagen):
    """
    Detecta el rostro en la imagen y predice la emoción
    """
    print("=" * 60)
    print("🎭 DETECTOR DE ROSTROS + EMOCIONES")
    print("=" * 60)
    
    # Cargar imagen
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"❌ Error: No se pudo cargar la imagen")
        return None
    
    # Cargar clasificador de rostros de OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convertir a grises para detección
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("❌ No se detectó ningún rostro. Usando imagen completa...")
        return predecir_emocion(ruta_imagen)
    
    print(f"✅ Detectados {len(faces)} rostro(s)")
    
    # Tomar el primer rostro
    (x, y, w, h) = faces[0]
    print(f"📐 Rostro en: x={x}, y={y}, w={w}, h={h}")
    
    # Recortar el rostro
    face_roi = img[y:y+h, x:x+w]
    
    # Guardar rostro temporalmente
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, face_roi)
    
    # Predecir emoción del rostro
    resultado = predecir_emocion(temp_path)
    
    # Eliminar archivo temporal
    os.remove(temp_path)
    
    # Dibujar rectángulo en la imagen original
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Mostrar emoción
    if resultado:
        cv2.putText(img, f"{resultado['emocion']}: {resultado['confianza']:.1f}%", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Mostrar imagen
    cv2.imshow('Emotion Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return resultado

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
if __name__ == "__main__":
    # Cambia esta ruta por la de tu imagen
    RUTA_IMAGEN = "tu_imagen.jpg"
    
    # Si no existe, usar imagen de prueba
    if not os.path.exists(RUTA_IMAGEN):
        print(f"⚠️ No se encuentra {RUTA_IMAGEN}")
        print("📸 Creando imagen de prueba...")
        
        # Crear una imagen de prueba (un rostro simulado)
        test_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Dibujar un círculo simulando un rostro
        cv2.circle(test_img, (100, 100), 80, (200, 150, 150), -1)
        cv2.circle(test_img, (70, 80), 15, (50, 50, 50), -1)  # ojo izquierdo
        cv2.circle(test_img, (130, 80), 15, (50, 50, 50), -1)  # ojo derecho
        cv2.ellipse(test_img, (100, 130), (40, 20), 0, 0, 180, (80, 50, 50), -2)  # boca
        cv2.imwrite("test_face.jpg", test_img)
        RUTA_IMAGEN = "test_face.jpg"
        print(f"✅ Creada: {RUTA_IMAGEN}")
    
    # Elegir método
    print("\n¿Cómo quieres procesar?")
    print("1. Usar imagen completa")
    print("2. Detectar rostro automáticamente")
    
    opcion = input("Elige (1 o 2): ").strip()
    
    if opcion == "2":
        resultado = detectar_y_predecir(RUTA_IMAGEN)
    else:
        resultado = predecir_emocion(RUTA_IMAGEN)
    
    if resultado:
        print("\n✅ Procesamiento completado")
    else:
        print("\n❌ Error en el procesamiento")