import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================================================
# REDUCIR MEMORIA
# ==================================================

mixed_precision.set_global_policy("mixed_float16")

# ==================================================
# DATASET
# ==================================================

dataset_path = "dataset/train"

emociones = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

X = []
y = []

print("======================================")
print("Cargando imágenes...")
print("======================================")

for idx, emocion in enumerate(emociones):

    carpeta = os.path.join(dataset_path, emocion)

    if not os.path.exists(carpeta):
        print(f"No existe {carpeta}")
        continue

    archivos = os.listdir(carpeta)

    print(f"{emocion}: {len(archivos)} imágenes")

    for archivo in archivos:

        ruta = os.path.join(carpeta, archivo)

        img = cv2.imread(ruta)

        if img is None:
            continue

        img = cv2.resize(img, (224,224))

        X.append(img)
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nTotal imágenes:", len(X))

# ==================================================
# PREPROCESAMIENTO
# ==================================================

X = preprocess_input(X)

# ==================================================
# TRAIN TEST
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# ==================================================
# DATA AUGMENTATION
# ==================================================

datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.20,

    height_shift_range=0.20,

    zoom_range=0.20,

    horizontal_flip=True,

    brightness_range=[0.8,1.2],

    fill_mode="nearest"

)

datagen.fit(X_train)

# ==================================================
# CLASS WEIGHTS
# ==================================================

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights")

print(class_weights)

# ==================================================
# MODELO
# ==================================================

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

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

model.summary()

# ==================================================
# CALLBACKS
# ==================================================

callbacks = [

    EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    ),

    ModelCheckpoint(

        "mejor_modelo.keras",

        monitor="val_accuracy",

        save_best_only=True

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        verbose=1

    )

]

# ==================================================
# PRIMER ENTRENAMIENTO
# ==================================================

print("\n===============================")
print("Entrenamiento inicial")
print("===============================\n")

history = model.fit(

    datagen.flow(

        X_train,

        y_train,

        batch_size=32

    ),

    validation_data=(X_test,y_test),

    epochs=15,

    class_weight=class_weights,

    callbacks=callbacks

)

# ==================================================
# FINE TUNING
# ==================================================

print("\n===============================")
print("Fine Tuning")
print("===============================\n")

base_model.trainable = True

for layer in base_model.layers[:-30]:

    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-5),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

history2 = model.fit(

    datagen.flow(

        X_train,

        y_train,

        batch_size=16

    ),

    validation_data=(X_test,y_test),

    epochs=10,

    class_weight=class_weights,

    callbacks=callbacks

)

# ==================================================
# CARGAR MEJOR MODELO
# ==================================================

model = load_model("mejor_modelo.keras")

model.save("modelo_resnet50_emociones.keras")

model.save("modelo_resnet50_emociones.h5")

print("\nModelo guardado correctamente")

# ==================================================
# EVALUACIÓN
# ==================================================

print("\nEvaluando modelo...")

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(

    y_test,

    y_pred

)

print("\nAccuracy:", accuracy)

print("\n")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=emociones

    )

)

# ==================================================
# MATRIZ DE CONFUSIÓN
# ==================================================

cm = confusion_matrix(

    y_test,

    y_pred

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=emociones

)

disp.plot(

    cmap="Blues",

    values_format="d"

)

plt.title("Matriz de Confusión ResNet50")

plt.show()

# ==================================================
# LIMPIAR MEMORIA
# ==================================================

del X
del y
del X_train
del X_test
del model

gc.collect()

tf.keras.backend.clear_session()

print("\nProceso terminado correctamente.")