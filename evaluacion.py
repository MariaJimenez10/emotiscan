from sklearn.metrics import classification_report
from emoci_deepface import DeepFace
import os

y_true = []
y_pred = []

carpeta = 'dataset'
emociones = ['Enojo', 'Felicidad', 'Neutral', 'Tristeza'] 

for emocion in os.listdir(carpeta):
    carpeta = os.path.join(carpeta, emocion)

    for img in os.listdir(carpeta):
        ruta_img = os.path.join(carpeta, img)

        try:
            resultado = DeepFace.analyze(ruta_img, actions=['emotion'], enforce_detection=False)
            pred = resultado[0]['dominant_emotion']

            y_true.append(emocion)
            y_pred.append(pred)

        except:
            continue

print(classification_report(y_true, y_pred))