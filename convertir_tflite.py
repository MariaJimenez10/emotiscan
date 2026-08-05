import tensorflow as tf
from tensorflow.keras.models import load_model

print("Cargando modelo...")

model = load_model(
    "modelo_resnet50_emociones.h5",
    compile=False
)

print("Convirtiendo a TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("modelo_resnet50_emociones.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversión terminada")