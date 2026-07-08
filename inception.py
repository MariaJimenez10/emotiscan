import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

# =========================
# RUTA DEL DATASET
# =========================

dataset_path = "dataset/train"

# =========================
# EMOCIONES
# =========================

emociones = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# =========================
# CARGAR DATOS
# =========================

X = []
y = []

print("Cargando imágenes...")

for idx, emocion in enumerate(emociones):

    carpeta = os.path.join(dataset_path, emocion)

    if not os.path.exists(carpeta):
        print(f"No existe la carpeta: {carpeta}")
        continue

    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        imagen = cv2.imread(ruta)

        if imagen is None:
            continue

        # InceptionV3 usa 299x299
        imagen = cv2.resize(imagen, (299, 299))

        X.append(imagen)
        y.append(idx)

X = np.array(X, dtype="float32")
y = np.array(y)

print("Total imágenes:", len(X))

# =========================
# PREPROCESAMIENTO INCEPTIONV3
# =========================

X = preprocess_input(X)

# =========================
# TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# MODELO INCEPTIONV3
# =========================

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    len(emociones),
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# =========================
# COMPILAR
# =========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# ENTRENAMIENTO
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# =========================
# GUARDAR MODELO
# =========================

model.save("modelo_inceptionv3_emociones.h5")

print("\n✅ Modelo guardado correctamente")
print("📁 Archivo: modelo_inceptionv3_emociones.h5")

# =========================
# PREDICCIONES
# =========================

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# =========================
# MÉTRICAS
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n========== RESULTADOS ==========")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# =========================
# REPORTE
# =========================

print("\n========== REPORTE ==========\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=emociones
    )
)

# =========================
# MATRIZ DE CONFUSIÓN
# =========================

matriz = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=emociones
)

disp.plot(cmap="Blues")

plt.title("Matriz de Confusión - InceptionV3")

plt.show()