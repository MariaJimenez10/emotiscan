import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from deepface import DeepFace

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# RUTA DATASET
# =========================

dataset_path = "dataset/test"

# =========================
# EMOCIONES
# =========================

emociones = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# Conversión DeepFace -> etiquetas del proyecto

mapa_emociones = {
    "angry": "Enojo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza"
}

# =========================
# LISTAS
# =========================

y_true = []
y_pred = []

print("Analizando imágenes con DeepFace...")

# =========================
# RECORRER DATASET
# =========================

for emocion_real in emociones:

    carpeta = os.path.join(
        dataset_path,
        emocion_real
    )

    if not os.path.exists(carpeta):
        continue

    for archivo in os.listdir(carpeta):

        ruta_imagen = os.path.join(
            carpeta,
            archivo
        )

        try:

            resultado = DeepFace.analyze(
                img_path=ruta_imagen,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            emocion_predicha = resultado[0]["dominant_emotion"]

            if emocion_predicha in mapa_emociones:

                y_true.append(emocion_real)

                y_pred.append(
                    mapa_emociones[emocion_predicha]
                )

        except Exception as e:

            print(
                f"Error en {archivo}: {e}"
            )

# =========================
# MÉTRICAS
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\n========== RESULTADOS DEEPFACE ==========\n")

print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score : {f1:.4f} ({f1*100:.2f}%)")

# =========================
# REPORTE
# =========================

print("\n========== REPORTE ==========\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)

# =========================
# MATRIZ DE CONFUSIÓN
# =========================

matriz = confusion_matrix(
    y_true,
    y_pred,
    labels=emociones
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=emociones
)

disp.plot(cmap="Blues")

plt.title(
    "Matriz de Confusión - DeepFace"
)

plt.show()