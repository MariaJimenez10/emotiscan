import tensorflow as tf
import numpy as np
import os

print("=" * 60)
print("🔍 VERIFICADOR DE MODELO TFLITE")
print("=" * 60)

# Ruta de tu modelo
model_path = "modelo_resnet50_emociones.tflite"

# Verificar que el archivo existe
if not os.path.exists(model_path):
    print(f"❌ ERROR: No se encuentra el archivo: {model_path}")
    print(f"📁 Directorio actual: {os.getcwd()}")
    print("\nArchivos en el directorio:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    exit()

print(f"✅ Modelo encontrado: {model_path}\n")

# Cargar el modelo
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("✅ Modelo cargado correctamente\n")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# Obtener detalles
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 60)
print("📥 INFORMACIÓN DE ENTRADA")
print("=" * 60)
print(f"Shape: {input_details[0]['shape']}")
print(f"Dtype: {input_details[0]['dtype']}")
print(f"Quantization: {input_details[0]['quantization']}")
print(f"Nombre: {input_details[0]['name']}")

print("\n" + "=" * 60)
print("📤 INFORMACIÓN DE SALIDA")
print("=" * 60)
print(f"Shape: {output_details[0]['shape']}")
print(f"Dtype: {output_details[0]['dtype']}")
print(f"Quantization: {output_details[0]['quantization']}")
print(f"Nombre: {output_details[0]['name']}")

print("\n" + "=" * 60)
print("📊 INTERPRETACIÓN")
print("=" * 60)

# Determinar formato esperado
expected_shape = input_details[0]['shape']
num_canales = expected_shape[-1]

if len(expected_shape) == 4:
    altura = expected_shape[1]
    ancho = expected_shape[2]
    
    if num_canales == 1:
        print(f"✅ El modelo espera imágenes en ESCALA DE GRISES")
        print(f"   Tamaño: {altura}x{ancho} pixels")
        print(f"   Formato: (batch, {altura}, {ancho}, 1)")
        print("\n💡 DEBES USAR:")
        print("   - Convertir a grises")
        print(f"   - Redimensionar a ({altura}, {ancho})")
        print("   - Normalizar /255.0")
        print("   - Shape final: (1, altura, ancho, 1)")
    else:
        print(f"✅ El modelo espera imágenes en COLOR (RGB)")
        print(f"   Tamaño: {altura}x{ancho} pixels")
        print(f"   Formato: (batch, {altura}, {ancho}, 3)")
        print("\n💡 DEBES USAR:")
        print("   - Mantener 3 canales (RGB)")
        print(f"   - Redimensionar a ({altura}, {ancho})")
        print("   - Usar preprocess_input de ResNet50 si aplica")
        print("   - Shape final: (1, altura, ancho, 3)")

# Verificar cuantización
quantization = input_details[0]['quantization']
if quantization[0] != 0:
    print(f"\n⚠️ EL MODELO ESTÁ CUANTIZADO")
    print(f"   Scale: {quantization[0]}")
    print(f"   Zero point: {quantization[1]}")
else:
    print("\n✅ Modelo sin cuantización (usa float32)")

print("\n" + "=" * 60)
print("🎯 PRUEBA RÁPIDA")
print("=" * 60)

# Crear entrada de prueba
test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)

# Si está cuantizado, convertir a int8
if quantization[0] != 0:
    test_input = np.zeros(input_details[0]['shape'], dtype=np.int8)

try:
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"✅ Prueba exitosa!")
    print(f"   Salida obtenida: {output.shape}")
    print(f"   Valores: {output[0][:4]}")  # Primeros 4 valores
    print(f"\n📊 NÚMERO DE CLASES: {output.shape[1]}")
except Exception as e:
    print(f"❌ Error en la prueba: {e}")

print("\n" + "=" * 60)
print("✅ VERIFICACIÓN COMPLETA")
print("=" * 60)