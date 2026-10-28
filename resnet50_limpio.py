
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# CONFIGURACIÓN SEGURA: NO REEMPLAZAR EL MODELO ACTUAL
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
NUM_CLASSES = len(CLASSES)

DATA_DIR = Path("dataset_limpio")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = Path("modelo")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet50_emociones_limpio.keras"
CLASS_PATH = MODEL_DIR / "clases_limpio.json"
REPORT_PATH = MODEL_DIR / "reporte_evaluacion_limpio.txt"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\n==========================================")
print(" EMOTISCAN - ENTRENAMIENTO EXPERIMENTAL")
print("==========================================")

# Protección: no sobrescribir resultados de una ejecución anterior.
if MODEL_PATH.exists():
    raise FileExistsError(
        f"Ya existe {MODEL_PATH}.\n"
        "No se sobrescribirá. Revisa ese archivo antes de continuar."
    )

# Protección: comprobar las carpetas antes de entrenar.
for carpeta in [TRAIN_DIR, TEST_DIR]:
    if not carpeta.is_dir():
        raise FileNotFoundError(
            f"No se encuentra la carpeta: {carpeta.resolve()}"
        )


def obtener_imagenes(carpeta):
    rutas = []
    etiquetas = []

    for indice, clase in enumerate(CLASSES):
        subcarpeta = carpeta / clase

        if not subcarpeta.is_dir():
            raise FileNotFoundError(
                f"Falta la carpeta: {subcarpeta}"
            )

        archivos = sorted(
            archivo
            for archivo in subcarpeta.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONS
        )

        if len(archivos) < 2:
            raise ValueError(
                f"No hay suficientes imágenes en {subcarpeta}"
            )

        rutas.extend(str(archivo.resolve()) for archivo in archivos)
        etiquetas.extend([indice] * len(archivos))

        print(f"{carpeta.name}/{clase}: {len(archivos)} imágenes")

    return np.array(rutas), np.array(etiquetas, dtype=np.int32)


print("\n--- REVISIÓN DEL ENTRENAMIENTO ---")
rutas_train, etiquetas_train = obtener_imagenes(TRAIN_DIR)

print("\n--- REVISIÓN DE LA PRUEBA ---")
rutas_test, etiquetas_test = obtener_imagenes(TEST_DIR)

# Separar una parte del entrenamiento para validar el aprendizaje.
rutas_fit, rutas_val, etiquetas_fit, etiquetas_val = train_test_split(
    rutas_train,
    etiquetas_train,
    test_size=0.15,
    random_state=SEED,
    stratify=etiquetas_train
)

print("\n--- DISTRIBUCIÓN ---")
print("Entrenamiento:", len(rutas_fit))
print("Validación:", len(rutas_val))
print("Prueba independiente del entrenamiento:", len(rutas_test))


def crear_generador(rutas, etiquetas, entrenamiento=False):
    if entrenamiento:
        generador = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=12,
            width_shift_range=0.10,
            height_shift_range=0.10,
            zoom_range=0.12,
            brightness_range=(0.85, 1.15),
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        generador = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    dataframe = pd.DataFrame({
        "filename": rutas,
        "class": [CLASSES[int(i)] for i in etiquetas]
    })

    return generador.flow_from_dataframe(
        dataframe=dataframe,
        x_col="filename",
        y_col="class",
        classes=CLASSES,
        target_size=IMG_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=entrenamiento,
        seed=SEED
    )


train_generator = crear_generador(
    rutas_fit, etiquetas_fit, entrenamiento=True
)
val_generator = crear_generador(
    rutas_val, etiquetas_val
)
test_generator = crear_generador(
    rutas_test, etiquetas_test
)

# Dar mayor peso a las clases menos representadas.
pesos = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=etiquetas_fit
)

class_weights = {
    i: float(pesos[i]) for i in range(NUM_CLASSES)
}

print("\n--- PESOS POR EMOCIÓN ---")
for i, clase in enumerate(CLASSES):
    print(f"{clase}: {class_weights[i]:.3f}")


# =========================================================
# CREAR EL MODELO NUEVO
# =========================================================

print("\nCargando ResNet50 con pesos de ImageNet...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.50)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

checkpoint = ModelCheckpoint(
    str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]


# =========================================================
# ETAPA 1: ENTRENAR LA CAPA DE CLASIFICACIÓN
# =========================================================

print("\n--- ETAPA 1: ENTRENAMIENTO INICIAL ---")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================================================
# ETAPA 2: AJUSTE FINO DE RESNET50
# =========================================================

print("\n--- ETAPA 2: FINE TUNING ---")

base_model.trainable = True

for layer in base_model.layers[:80]:
    layer.trainable = False

# Mantener las capas BatchNormalization congeladas.
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks
)


# =========================================================
# EVALUACIÓN FINAL: NO UTILIZAR TEST PARA ENTRENAR
# =========================================================

print("\n--- CARGANDO EL MEJOR MODELO NUEVO ---")

model = tf.keras.models.load_model(str(MODEL_PATH))

print("\n--- EVALUACIÓN FINAL EN TEST ---")

test_generator.reset()

loss, accuracy = model.evaluate(test_generator, verbose=1)

test_generator.reset()
probabilidades = model.predict(test_generator, verbose=1)

y_pred = np.argmax(probabilidades, axis=1)
y_true = test_generator.classes

reporte = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

matriz = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

print("\nAccuracy de prueba:", f"{accuracy * 100:.2f}%")
print("\nREPORTE POR EMOCIÓN")
print(reporte)

print("\nMATRIZ DE CONFUSIÓN")
print("Orden:", CLASSES)
print(matriz)

print("\nRECALL POR EMOCIÓN")
for i, clase in enumerate(CLASSES):
    total = int(matriz[i].sum())
    recall = matriz[i, i] / total if total else 0.0
    print(
        f"{clase}: {recall * 100:.2f}% "
        f"({matriz[i, i]}/{total})"
    )


# Guardar los resultados con nombres nuevos.
with REPORT_PATH.open("w", encoding="utf-8") as archivo:
    archivo.write(f"Accuracy test: {accuracy * 100:.2f}%\n\n")
    archivo.write("Orden de clases: " + ", ".join(CLASSES) + "\n\n")
    archivo.write(reporte + "\n")
    archivo.write("Matriz de confusión:\n")
    archivo.write(str(matriz) + "\n")

with CLASS_PATH.open("w", encoding="utf-8") as archivo:
    json.dump(CLASSES, archivo, ensure_ascii=False, indent=4)

print("\n--- ENTRENAMIENTO TERMINADO ---")
print("Modelo experimental:", MODEL_PATH)
print("Clases:", CLASS_PATH)
print("Reporte:", REPORT_PATH)
print("\nEl modelo anterior y la aplicación Flask no se modificaron.")
