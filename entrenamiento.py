import cv2
import os
import numpy as np

dataPath = 'dataset'
emotions = os.listdir(dataPath)

print('Emociones:', emotions)

labels = []
facesData = []
label = 0

for emotion in emotions:
    emotionPath = os.path.join(dataPath, emotion)

    if os.path.isdir(emotionPath):
        for file in os.listdir(emotionPath):
            imgPath = os.path.join(emotionPath, file)
            img = cv2.imread(imgPath, 0)

            if img is not None:
                facesData.append(img)
                labels.append(label)

        label += 1

print("Entrenando modelo...")

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer.train(facesData, np.array(labels))

emotion_recognizer.write("modelo_emociones.xml")

print("Modelo entrenado correctamente")