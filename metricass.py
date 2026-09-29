import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIGURACIÓN
# =========================

MODELO = "modelo/mejor_modelo.keras"
DATASET_TEST = "dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]

# =========================
# CARGAR MODELO
# =========================

print("Cargando modelo...")

modelo = tf.keras.models.load_model(MODELO)

print("Modelo cargado correctamente.")

# =========================
# CARGAR DATASET DE PRUEBA
# =========================

datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = datagen.flow_from_directory(
    DATASET_TEST,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClases detectadas:")
print(test_generator.class_indices)

# =========================
# EVALUACIÓN
# =========================

print("\nEvaluando modelo...")

resultado = modelo.evaluate(
    test_generator,
    verbose=1
)

print("\n==============================")
print("RESULTADOS DEL MODELO")
print("==============================")

for nombre, valor in zip(modelo.metrics_names, resultado):
    print(f"{nombre}: {valor:.4f}")

# =========================
# PREDICCIONES
# =========================

print("\nGenerando predicciones...")

predicciones = modelo.predict(
    test_generator,
    verbose=1
)

y_pred = np.argmax(predicciones, axis=1)
y_real = test_generator.classes

# =========================
# REPORTE DE CLASIFICACIÓN
# =========================

nombres_clases = list(test_generator.class_indices.keys())

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_real,
        y_pred,
        target_names=nombres_clases,
        digits=4
    )
)

# =========================
# MATRIZ DE CONFUSIÓN
# =========================

matriz = confusion_matrix(y_real, y_pred)

print("\n==============================")
print("MATRIZ DE CONFUSIÓN")
print("==============================")

print(matriz)

# =========================
# GUARDAR RESULTADOS
# =========================

np.savetxt(
    "matriz_confusion.txt",
    matriz,
    fmt="%d"
)

print("\nMatriz de confusión guardada en:")
print("matriz_confusion.txt")