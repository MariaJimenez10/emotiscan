import os
import random
import json

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


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
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet50_emociones_mejorado.keras"
)

TFLITE_PATH = os.path.join(
    MODEL_DIR,
    "modelo_resnet50_emociones.tflite"
)

CLASSES_PATH = os.path.join(
    MODEL_DIR,
    "clases.json"
)

# ==========================================================
# MUY IMPORTANTE:
# ESTE ORDEN DEBE SER EL MISMO EN ENTRENAMIENTO E INFERENCIA
# ==========================================================

CLASSES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

NUM_CLASSES = len(CLASSES)


# ==========================================================
# VERIFICAR CARPETAS
# ==========================================================

print("\n" + "=" * 60)
print("VERIFICANDO DATASET")
print("=" * 60)

for clase in CLASSES:

    train_class_dir = os.path.join(TRAIN_DIR, clase)
    test_class_dir = os.path.join(TEST_DIR, clase)

    if not os.path.isdir(train_class_dir):
        raise FileNotFoundError(
            f"No existe la carpeta de entrenamiento: {train_class_dir}"
        )

    if not os.path.isdir(test_class_dir):
        raise FileNotFoundError(
            f"No existe la carpeta de prueba: {test_class_dir}"
        )

    print(f"✅ {clase}")


# ==========================================================
# GENERADOR DE ENTRENAMIENTO
# ==========================================================
#
# Usamos SOLO el 80% de TRAIN para entrenar.
#
# El 20% restante será VALIDACIÓN.
#
# El TEST queda completamente separado y solo se utiliza
# al final para medir el modelo.
# ==========================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.08,

    brightness_range=(0.90, 1.10),

    horizontal_flip=True,

    fill_mode="nearest",

    validation_split=0.20
)


# ==========================================================
# GENERADOR DE VALIDACIÓN
# ==========================================================
#
# IMPORTANTE:
# La validación NO tendrá aumentos artificiales.
# Solo se aplica preprocess_input.
# ==========================================================

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)


# ==========================================================
# TRAIN
# ==========================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,

    classes=CLASSES,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=SEED
)


# ==========================================================
# VALIDACIÓN
# ==========================================================

val_generator = val_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,

    classes=CLASSES,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=SEED
)


# ==========================================================
# TEST
# ==========================================================
#
# ESTE DATASET NO SE UTILIZA DURANTE EL ENTRENAMIENTO.
#
# Solo se utiliza al final para conocer el rendimiento real.
# ==========================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


test_generator = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,

    classes=CLASSES,

    class_mode="categorical",

    shuffle=False
)


# ==========================================================
# MOSTRAR INFORMACIÓN DEL DATASET
# ==========================================================

print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)

print(f"Imágenes entrenamiento : {train_generator.samples}")
print(f"Imágenes validación    : {val_generator.samples}")
print(f"Imágenes prueba        : {test_generator.samples}")

print("\nÍndices de las clases:")

for clase, indice in train_generator.class_indices.items():
    print(f"  {indice} -> {clase}")


# ==========================================================
# CONTAR IMÁGENES POR CLASE
# ==========================================================

print("\n" + "=" * 60)
print("IMÁGENES POR CLASE EN TRAIN")
print("=" * 60)

train_labels = train_generator.classes

for indice, clase in enumerate(CLASSES):

    cantidad = np.sum(train_labels == indice)

    print(
        f"{clase:<12}: {cantidad}"
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================
#
# Esto ayuda a que las clases con menos imágenes no sean
# ignoradas por el modelo.
#
# NO estamos forzando ninguna emoción.
#
# Simplemente compensamos el desequilibrio del dataset.
# ==========================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = {
    indice: float(peso)
    for indice, peso in enumerate(class_weights_array)
}


print("\n" + "=" * 60)
print("PESOS DE LAS CLASES")
print("=" * 60)

for indice, clase in enumerate(CLASSES):

    print(
        f"{clase:<12}: {class_weights[indice]:.4f}"
    )


# ==========================================================
# GUARDAR MAPEO DE CLASES
# ==========================================================
#
# Esto nos ayudará después a garantizar que predict.py
# utilice exactamente el mismo orden.
# ==========================================================

with open(CLASSES_PATH, "w", encoding="utf-8") as archivo:

    json.dump(
        CLASSES,
        archivo,
        ensure_ascii=False,
        indent=4
    )

print(f"\n✅ Clases guardadas en: {CLASSES_PATH}")


# ==========================================================
# MODELO RESNET50
# ==========================================================

print("\n" + "=" * 60)
print("CREANDO RESNET50")
print("=" * 60)

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# ==========================================================
# PRIMERA ETAPA:
# CONGELAR RESNET50
# ==========================================================

base_model.trainable = False


# ==========================================================
# CAPA DE ENTRADA
# ==========================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)


# ==========================================================
# RESNET50
# ==========================================================
#
# training=False mantiene BatchNormalization estable.
# ==========================================================

x = base_model(
    inputs,
    training=False
)


# ==========================================================
# CLASIFICADOR
# ==========================================================

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.35
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="emociones"
)(x)


# ==========================================================
# CREAR MODELO
# ==========================================================

model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================================
# COMPILACIÓN ETAPA 1
# ==========================================================

model.compile(
    optimizer=Adam(
        learning_rate=3e-4
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ==========================================================
# MOSTRAR MODELO
# ==========================================================

model.summary()


# ==========================================================
# CALLBACKS ETAPA 1
# ==========================================================

checkpoint_stage1 = ModelCheckpoint(
    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping_stage1 = EarlyStopping(
    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr_stage1 = ReduceLROnPlateau(
    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ==========================================================
# ETAPA 1
# ==========================================================

print("\n" + "=" * 60)
print("ETAPA 1: ENTRENAMIENTO CON RESNET50 CONGELADA")
print("=" * 60)

history_stage1 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=15,

    class_weight=class_weights,

    callbacks=[
        checkpoint_stage1,
        early_stopping_stage1,
        reduce_lr_stage1
    ]
)


# ==========================================================
# CARGAR MEJOR MODELO DE ETAPA 1
# ==========================================================

print("\nCargando mejor modelo de la etapa 1...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================================
# ETAPA 2:
# FINE-TUNING
# ==========================================================

print("\n" + "=" * 60)
print("ETAPA 2: FINE-TUNING")
print("=" * 60)


# Activamos el entrenamiento de ResNet50

base_model = model.layers[1]

base_model.trainable = True


# ==========================================================
# CONGELAR CASI TODA LA RESNET50
# ==========================================================
#
# Solo vamos a entrenar las últimas capas.
#
# Esto permite adaptar ResNet50 a expresiones faciales sin
# destruir las características aprendidas con ImageNet.
# ==========================================================

fine_tune_layers = 40

fine_tune_at = max(
    0,
    len(base_model.layers) - fine_tune_layers
)

print(
    f"Total capas ResNet50: {len(base_model.layers)}"
)

print(
    f"Se entrenarán desde la capa: {fine_tune_at}"
)


for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False


for layer in base_model.layers[fine_tune_at:]:

    layer.trainable = True


# ==========================================================
# BATCH NORMALIZATION
# ==========================================================
#
# Las dejamos congeladas para evitar inestabilidad.
# ==========================================================

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


# ==========================================================
# RECOMPILAR CON LR PEQUEÑO
# ==========================================================

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ==========================================================
# CALLBACKS ETAPA 2
# ==========================================================

checkpoint_stage2 = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping_stage2 = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr_stage2 = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ==========================================================
# ENTRENAMIENTO ETAPA 2
# ==========================================================

history_stage2 = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=20,

    class_weight=class_weights,

    callbacks=[
        checkpoint_stage2,
        early_stopping_stage2,
        reduce_lr_stage2
    ]
)


# ==========================================================
# CARGAR EL MEJOR MODELO FINAL
# ==========================================================

print("\n" + "=" * 60)
print("CARGANDO MEJOR MODELO")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================================
# EVALUACIÓN FINAL
# ==========================================================
#
# AHORA SÍ usamos TEST.
#
# Nunca se utilizó durante el entrenamiento.
# ==========================================================

print("\n" + "=" * 60)
print("EVALUACIÓN FINAL CON TEST")
print("=" * 60)


test_generator.reset()


test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)


print(
    f"\nAccuracy final: {test_accuracy:.4f}"
)


# ==========================================================
# PREDICCIONES
# ==========================================================

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


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

print("\n" + "=" * 60)
print("MATRIZ DE CONFUSIÓN")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred
)


print(
    "Filas = emoción REAL"
)

print(
    "Columnas = emoción PREDICHA"
)

print()

print(
    "             "
    + " ".join(
        f"{clase:>12}"
        for clase in CLASSES
    )
)


for i, clase in enumerate(CLASSES):

    fila = " ".join(
        f"{cm[i][j]:>12}"
        for j in range(NUM_CLASSES)
    )

    print(
        f"{clase:<12}{fila}"
    )


# ==========================================================
# RECALL POR EMOCIÓN
# ==========================================================

print("\n" + "=" * 60)
print("RECALL POR EMOCIÓN")
print("=" * 60)

for i, clase in enumerate(CLASSES):

    total_real = np.sum(
        y_true == i
    )

    correctas = cm[i][i]

    if total_real > 0:

        recall = correctas / total_real

    else:

        recall = 0.0

    print(
        f"{clase:<12}: {recall:.4f}"
    )


# ==========================================================
# GUARDAR MODELO TFLITE
# ==========================================================

print("\n" + "=" * 60)
print("CONVIRTIENDO A TFLITE")
print("=" * 60)

try:

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    tflite_model = converter.convert()

    with open(
        TFLITE_PATH,
        "wb"
    ) as archivo:

        archivo.write(
            tflite_model
        )

    print(
        f"✅ TFLite creado correctamente:"
    )

    print(
        TFLITE_PATH
    )

except Exception as e:

    print(
        "❌ Error convirtiendo a TFLite:"
    )

    print(e)


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 60)
print("ENTRENAMIENTO TERMINADO")
print("=" * 60)

print(
    f"✅ Modelo Keras:"
)

print(
    MODEL_PATH
)

print(
    f"\n✅ Modelo TFLite:"
)

print(
    TFLITE_PATH
)

print(
    f"\n✅ Clases:"
)

for i, clase in enumerate(CLASSES):

    print(
        f"   {i} -> {clase}"
    )

print("\n")