import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

MODEL_DIR = "modelo"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet50_emociones_mejorado.keras"
)

# IMPORTANTE:
# Este orden DEBE mantenerse también en predict.py
CLASSES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

NUM_CLASSES = len(CLASSES)

print("=" * 70)
print("ENTRENAMIENTO RESNET50 - EMOTISCAN")
print("=" * 70)

print("\nClases:")
for i, clase in enumerate(CLASSES):
    print(f"  {i} = {clase}")

print(f"\nImagen: {IMG_SIZE}")
print(f"Batch: {BATCH_SIZE}")
print(f"Train: {TRAIN_DIR}")
print(f"Test:  {TEST_DIR}")


# ============================================================
# GENERADORES
# ============================================================

print("\n" + "=" * 70)
print("PREPARANDO DATASET")
print("=" * 70)


# AUMENTO DE DATOS
#
# Se aplican transformaciones para que el modelo no memorice
# solamente las imágenes del dataset.
#
# No usamos cambios demasiado agresivos porque podrían alterar
# las características emocionales del rostro.

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=12,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,

    brightness_range=(0.85, 1.15),

    horizontal_flip=True,

    fill_mode="nearest"
)


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    classes=CLASSES,

    class_mode="categorical",

    shuffle=True,

    seed=SEED
)


test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    classes=CLASSES,

    class_mode="categorical",

    shuffle=False
)


print("\n" + "=" * 70)
print("CLASES DETECTADAS")
print("=" * 70)

print(train_generator.class_indices)

print("\nImágenes entrenamiento:", train_generator.samples)
print("Imágenes prueba:", test_generator.samples)


# ============================================================
# CALCULAR PESOS DE LAS CLASES
# ============================================================

print("\n" + "=" * 70)
print("CALCULANDO PESOS DE CLASE")
print("=" * 70)


class_counts = np.bincount(
    train_generator.classes,
    minlength=NUM_CLASSES
)

print("\nCantidad de imágenes por clase:")

for i, clase in enumerate(CLASSES):
    print(f"{i} - {clase}: {class_counts[i]}")


classes_presentes = np.arange(NUM_CLASSES)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes_presentes,
    y=train_generator.classes
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}


print("\nPesos calculados:")

for i, clase in enumerate(CLASSES):
    print(
        f"{clase}: {class_weights[i]:.4f}"
    )


# ============================================================
# MODELO RESNET50
# ============================================================

print("\n" + "=" * 70)
print("CREANDO RESNET50")
print("=" * 70)


base_model = ResNet50(
    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# Primera etapa:
# congelamos ResNet50

base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


x = base_model(
    inputs,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.BatchNormalization()(x)


x = layers.Dropout(
    0.45
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILACIÓN ETAPA 1
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_loss",

    patience=6,

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


# ============================================================
# ETAPA 1
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 1 - ENTRENAMIENTO CON RESNET50 CONGELADA")
print("=" * 70)


history1 = model.fit(
    train_generator,

    validation_data=test_generator,

    epochs=12,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# ETAPA 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 70)
print("ETAPA 2 - FINE TUNING")
print("=" * 70)


# Descongelamos ResNet50

base_model.trainable = True


# Primero congelamos las primeras capas.
#
# Dejamos entrenables las capas superiores,
# que son las que más nos interesa adaptar
# al reconocimiento de emociones.

fine_tune_from = 100


for layer in base_model.layers[:fine_tune_from]:

    layer.trainable = False


for layer in base_model.layers[fine_tune_from:]:

    layer.trainable = True


# BatchNormalization puede generar inestabilidad
# durante fine tuning con datasets pequeños.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


print(
    f"\nCapas congeladas inicialmente: "
    f"{fine_tune_from}"
)


trainable_count = sum(
    1 for layer in model.layers
    if layer.trainable
)

print(
    "Capas entrenables:",
    trainable_count
)


# Learning rate MUCHO menor para fine tuning.

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# Nuevos callbacks para fine tuning.

checkpoint_ft = ModelCheckpoint(
    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping_ft = EarlyStopping(
    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


reduce_lr_ft = ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-8,

    verbose=1
)


history2 = model.fit(
    train_generator,

    validation_data=test_generator,

    epochs=20,

    class_weight=class_weights,

    callbacks=[
        checkpoint_ft,
        early_stopping_ft,
        reduce_lr_ft
    ]
)


# ============================================================
# CARGAR MEJOR MODELO
# ============================================================

print("\n" + "=" * 70)
print("CARGANDO MEJOR MODELO")
print("=" * 70)


model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# EVALUACIÓN
# ============================================================

print("\n" + "=" * 70)
print("EVALUACIÓN FINAL")
print("=" * 70)


test_generator.reset()


loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print("\nLoss:", loss)
print("Accuracy:", accuracy)


# ============================================================
# PREDICCIONES
# ============================================================

print("\nGenerando predicciones...")


test_generator.reset()


predicciones = model.predict(
    test_generator,
    verbose=1
)


y_pred = np.argmax(
    predicciones,
    axis=1
)

y_true = test_generator.classes


# ============================================================
# MÉTRICAS
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


print("\n" + "=" * 70)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 70)


reporte = classification_report(
    y_true,
    y_pred,

    target_names=CLASSES,

    digits=4,

    zero_division=0
)


print(reporte)


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

matriz = confusion_matrix(
    y_true,
    y_pred
)


print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN")
print("=" * 70)


print("\nFilas = REAL")
print("Columnas = PREDICCIÓN\n")


print(" " * 15, end="")

for clase in CLASSES:
    print(
        f"{clase:>12}",
        end=""
    )

print()


for i, fila in enumerate(matriz):

    print(
        f"{CLASSES[i]:>15}",
        end=""
    )

    for valor in fila:

        print(
            f"{valor:>12}",
            end=""
        )

    print()


# ============================================================
# RECALL INDIVIDUAL
# ============================================================

print("\n" + "=" * 70)
print("RECALL POR EMOCIÓN")
print("=" * 70)


for i, clase in enumerate(CLASSES):

    verdaderos = matriz[i, i]

    total_reales = np.sum(
        matriz[i]
    )

    if total_reales > 0:

        recall = (
            verdaderos /
            total_reales
        )

    else:

        recall = 0


    print(
        f"{clase:12}: "
        f"{recall * 100:.2f}%"
    )


# ============================================================
# GUARDAR MODELO FINAL
# ============================================================

print("\n" + "=" * 70)
print("MODELO GUARDADO")
print("=" * 70)


print(
    "\nArchivo:"
)

print(
    MODEL_PATH
)


print("\n" + "=" * 70)
print("ENTRENAMIENTO TERMINADO")
print("=" * 70)