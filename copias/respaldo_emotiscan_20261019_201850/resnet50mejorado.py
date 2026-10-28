import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_emociones_mejorado.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# IMPORTANTE:
# ESTE ORDEN DEBE SER IGUAL AL DE LAS CARPETAS
CLASSES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

NUM_CLASSES = len(CLASSES)

print("\n==========================================")
print("      ENTRENAMIENTO EMOTISCAN")
print("==========================================")
print("Clases:", CLASSES)
print("==========================================\n")


# ==========================================================
# GENERADORES
# ==========================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.15,

    brightness_range=(0.75, 1.25),

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


print("\n==========================================")
print("CLASES DETECTADAS")
print("==========================================")

print(train_generator.class_indices)

print("\nImágenes entrenamiento:", train_generator.samples)
print("Imágenes prueba:", test_generator.samples)


# ==========================================================
# PESOS DE CLASE
# ==========================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_generator.classes
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\n==========================================")
print("PESOS DE CLASE")
print("==========================================")

for i, clase in enumerate(CLASSES):
    print(
        f"{clase}: {class_weights[i]:.3f}"
    )


# ==========================================================
# MODELO RESNET50
# ==========================================================

print("\n==========================================")
print("CARGANDO RESNET50")
print("==========================================\n")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


inputs = Input(shape=(224, 224, 3))

x = base_model(
    inputs,
    training=False
)

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(0.50)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)


# ==========================================================
# ETAPA 1
# ==========================================================

print("\n==========================================")
print("ETAPA 1 - ENTRENAMIENTO")
print("==========================================\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=2e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
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


history1 = model.fit(
    train_generator,

    validation_data=test_generator,

    epochs=15,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ==========================================================
# ETAPA 2 - FINE TUNING
# ==========================================================

print("\n==========================================")
print("ETAPA 2 - FINE TUNING")
print("==========================================\n")

base_model.trainable = True


# Congelar primeras capas
for layer in base_model.layers[:80]:
    layer.trainable = False


# Mantener BatchNormalization congelado
for layer in base_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history2 = model.fit(
    train_generator,

    validation_data=test_generator,

    epochs=25,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ==========================================================
# CARGAR MEJOR MODELO
# ==========================================================

print("\n==========================================")
print("CARGANDO MEJOR MODELO")
print("==========================================\n")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================================
# EVALUACIÓN
# ==========================================================

print("\n==========================================")
print("EVALUACIÓN FINAL")
print("==========================================\n")

test_generator.reset()

loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\nAccuracy final:")
print(f"{accuracy * 100:.2f}%")


# ==========================================================
# PREDICCIONES
# ==========================================================

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


# ==========================================================
# REPORTE
# ==========================================================

print("\n==========================================")
print("REPORTE DE CLASIFICACIÓN")
print("==========================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4
    )
)


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

print("\n==========================================")
print("MATRIZ DE CONFUSIÓN")
print("==========================================\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("              ", CLASSES)

for i, fila in enumerate(cm):
    print(
        f"{CLASSES[i]:12} {fila}"
    )


# ==========================================================
# RECALL POR EMOCIÓN
# ==========================================================

print("\n==========================================")
print("RECALL POR EMOCIÓN")
print("==========================================\n")

for i, clase in enumerate(CLASSES):

    verdaderos = cm[i, i]

    total = np.sum(cm[i])

    recall = (
        verdaderos / total
        if total > 0
        else 0
    )

    print(
        f"{clase}: {recall * 100:.2f}%"
    )


print("\n==========================================")
print("ENTRENAMIENTO TERMINADO")
print("==========================================")

print("\nModelo guardado en:")

print(MODEL_PATH)

print("\n==========================================\n")
