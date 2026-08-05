import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# REDUCIR MEMORIA
# =========================

mixed_precision.set_global_policy("mixed_float16")

# =========================
# DATASET
# =========================

dataset_path = "dataset/train"

emociones = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

X = []
y = []

print("Cargando imágenes...")

for idx, emocion in enumerate(emociones):

    carpeta = os.path.join(dataset_path, emocion)

    if not os.path.exists(carpeta):
        continue

    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        img = cv2.imread(ruta)

        if img is None:
            continue

        img = cv2.resize(img, (224,224))

        X.append(img)
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Imágenes:", len(X))

X = preprocess_input(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# MODELO
# =========================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    len(emociones),
    activation="softmax",
    dtype="float32"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "mejor_modelo.keras",
        save_best_only=True
    )

]

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test,y_test),

    epochs=15,

    batch_size=32,

    callbacks=callbacks

)

print("Guardando modelo...")

model.save("modelo_resnet50_emociones.keras")

model.save("modelo_resnet50_emociones.h5")

print("Modelo guardado.")

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test,y_pred))

print(classification_report(
    y_test,
    y_pred,
    target_names=emociones
))

cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emociones
)

disp.plot(cmap="Blues")

plt.show()

del X
del y
del X_train
del X_test
del model

gc.collect()

tf.keras.backend.clear_session()