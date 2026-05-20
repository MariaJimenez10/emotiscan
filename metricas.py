import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# =========================
# RUTA DEL DATASET
# =========================

dataset_path = "dataset/train"

# =========================
# EMOCIONES
# =========================


emociones = ['Enojo', 'Felicidad', 'Neutral', 'Tristeza']

# =========================
# VARIABLES
# =========================

X = []
y = []

# =========================
# CARGAR IMÁGENES
# =========================

for idx, emocion in enumerate(emociones):

    carpeta = os.path.join(dataset_path, emocion)

    for archivo in os.listdir(carpeta):

        ruta_imagen = os.path.join(carpeta, archivo)

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            continue

        # Redimensionar imagen
        imagen = cv2.resize(imagen, (48, 48))

        # Guardar imagen
        X.append(imagen)

        # Guardar etiqueta
        y.append(idx)

# =========================
# CONVERTIR A NUMPY
# =========================

X = np.array(X)
y = np.array(y)

# =========================
# NORMALIZAR
# =========================

X = X / 255.0

# =========================
# SEPARAR DATOS
# 80% entrenamiento
# 20% prueba
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# CREAR MODELO
# =========================

model = Sequential([
    Flatten(input_shape=(48, 48, 3)),
    Dense(128, activation='relu'),
    Dense(len(emociones), activation='softmax')
])

# =========================
# COMPILAR MODELO
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# ENTRENAR MODELO
# =========================

model.fit(
    X_train,
    y_train,
    epochs=5
)

# =========================
# PREDICCIONES
# =========================

y_pred = model.predict(X_test)

# Convertir probabilidades
# a clases reales

y_pred = np.argmax(y_pred, axis=1)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# =========================
# PRECISION
# =========================

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

print("Precision:", precision)

# =========================
# RECALL
# =========================

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

print("Recall:", recall)

# =========================
# F1 SCORE
# =========================

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("F1-Score:", f1)

# =========================
# REPORTE COMPLETO
# =========================

print("\nReporte Completo:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=emociones
))

# =========================
# MATRIZ DE CONFUSIÓN
# =========================

matriz = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=emociones
)

disp.plot(cmap='Blues')

plt.show()