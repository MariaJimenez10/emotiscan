import cv2
import os
import numpy as np

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

# Carpeta donde están las imágenes de entrenamiento
dataPath = 'dataset/train'

# Las 4 emociones que utilizará el modelo
emotions = ['Enojo', 'Felicidad', 'Neutral', 'Tristeza']

print("==========================================")
print("   ENTRENAMIENTO DEL RECONOCEDOR")
print("==========================================")
print("Carpeta de entrenamiento:", dataPath)
print("Emociones:", emotions)
print("------------------------------------------")


# ==========================================================
# CARGAR IMÁGENES Y ETIQUETAS
# ==========================================================

labels = []
facesData = []

# Recorremos las 4 emociones
for label, emotion in enumerate(emotions):

    emotionPath = os.path.join(dataPath, emotion)

    # Verificar que la carpeta exista
    if not os.path.isdir(emotionPath):
        print(f"⚠️ No existe la carpeta: {emotionPath}")
        continue

    print(f"\nProcesando emoción: {emotion}")
    print(f"Carpeta: {emotionPath}")

    image_count = 0

    # Recorrer imágenes de la emoción
    for file in os.listdir(emotionPath):

        imgPath = os.path.join(emotionPath, file)

        # Leer imagen en escala de grises
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

        if img is not None:

            facesData.append(img)
            labels.append(label)

            image_count += 1

    print(f"Imágenes cargadas: {image_count}")


# ==========================================================
# VERIFICAR DATOS
# ==========================================================

print("\n==========================================")
print("   RESUMEN DEL DATASET")
print("==========================================")

print("Total de imágenes:", len(facesData))
print("Total de etiquetas:", len(labels))

if len(facesData) == 0:
    print("❌ No se encontraron imágenes para entrenar.")
    exit()

print("\nDistribución de etiquetas:")

for label, emotion in enumerate(emotions):
    cantidad = labels.count(label)
    print(f"{label} -> {emotion}: {cantidad} imágenes")


# ==========================================================
# CREAR RECONOCEDOR LBPH
# ==========================================================

print("\n==========================================")
print("   CREANDO MODELO LBPH")
print("==========================================")

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()


# ==========================================================
# ENTRENAR MODELO
# ==========================================================

print("Entrenando modelo...")

emotion_recognizer.train(
    facesData,
    np.array(labels)
)
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
print(img.shape)

# ==========================================================
# GUARDAR MODELO
# ==========================================================

modelPath = "modelo_emociones.xml"

emotion_recognizer.write(modelPath)

print("\n==========================================")
print("   ENTRENAMIENTO COMPLETADO")
print("==========================================")

print(f"✅ Modelo guardado en: {modelPath}")

print("\nEmociones aprendidas por el modelo:")

for label, emotion in enumerate(emotions):
    print(f"{label} -> {emotion}")

print("==========================================")