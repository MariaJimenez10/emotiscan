import os
import json
import logging
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "modelo"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "resultados"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# CLASES
# ==========================================================
#
# ESTE ORDEN ES MUY IMPORTANTE.
#
# 0 = Enojo
# 1 = Felicidad
# 2 = Tristeza
# 3 = Neutral
#
# Este mismo orden deberá utilizarse después
# en predict.py y en Render.
# ==========================================================

CLASSES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

NUM_CLASSES = len(CLASSES)


# ==========================================================
# PARÁMETROS
# ==========================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16

VALIDATION_SPLIT = 0.20

SEED = 42

INITIAL_EPOCHS = 15

FINE_TUNE_EPOCHS = 15

TOTAL_EPOCHS = (
    INITIAL_EPOCHS +
    FINE_TUNE_EPOCHS
)


# ==========================================================
# SEMILLAS
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ==========================================================
# CONFIGURACIÓN CPU / GPU
# ==========================================================

print()
print("=" * 70)
print("              CONFIGURACIÓN DEL ENTORNO")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("✅ GPU DETECTADA")

    print(gpus)

else:

    print("⚠️ NO SE DETECTÓ GPU")

    print("Se utilizará CPU.")


# ==========================================================
# INFORMACIÓN DEL ENTRENAMIENTO
# ==========================================================

print()
print("=" * 70)
print("              ENTRENAMIENTO EMOTISCAN")
print("=" * 70)

print()

print("Clases:")

for i, emotion in enumerate(CLASSES):

    print(
        f"{i} -> {emotion}"
    )

print()

print(
    f"Tamaño de imagen: "
    f"{IMG_SIZE[0]} x {IMG_SIZE[1]}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Validación: {VALIDATION_SPLIT * 100:.0f}%"
)

print(
    f"Épocas iniciales: {INITIAL_EPOCHS}"
)

print(
    f"Épocas fine-tuning: {FINE_TUNE_EPOCHS}"
)


# ==========================================================
# VERIFICAR DATASET
# ==========================================================

print()
print("=" * 70)
print("              VERIFICANDO DATASET")
print("=" * 70)


if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta:\n{DATASET_DIR}"
    )


if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta:\n{TRAIN_DIR}"
    )


if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta:\n{TEST_DIR}"
    )


# ==========================================================
# VERIFICAR CLASES
# ==========================================================

for class_name in CLASSES:

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    test_class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.exists(
        train_class_dir
    ):

        raise FileNotFoundError(
            f"Falta la carpeta:\n"
            f"{train_class_dir}"
        )

    if not os.path.exists(
        test_class_dir
    ):

        raise FileNotFoundError(
            f"Falta la carpeta:\n"
            f"{test_class_dir}"
        )


print(
    "✅ Todas las carpetas de clases existen."
)


# ==========================================================
# CONTAR IMÁGENES
# ==========================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def count_images(folder):

    if not os.path.exists(folder):

        return 0

    total = 0

    for filename in os.listdir(folder):

        if filename.lower().endswith(
            VALID_EXTENSIONS
        ):

            filepath = os.path.join(
                folder,
                filename
            )

            if os.path.isfile(filepath):

                total += 1

    return total


# ==========================================================
# MOSTRAR DISTRIBUCIÓN
# ==========================================================

print()
print("=" * 70)
print("              DISTRIBUCIÓN DEL DATASET")
print("=" * 70)

total_train = 0
total_test = 0

for class_name in CLASSES:

    train_count = count_images(
        os.path.join(
            TRAIN_DIR,
            class_name
        )
    )

    test_count = count_images(
        os.path.join(
            TEST_DIR,
            class_name
        )
    )

    total_train += train_count
    total_test += test_count

    print(
        f"{class_name:12s} | "
        f"Train: {train_count:5d} | "
        f"Test: {test_count:5d}"
    )


print("-" * 70)

print(
    f"TOTAL TRAIN: {total_train}"
)

print(
    f"TOTAL TEST : {total_test}"
)

print(
    f"TOTAL DATASET: "
    f"{total_train + total_test}"
)


# ==========================================================
# GENERADOR DE ENTRENAMIENTO
# ==========================================================

print()
print("=" * 70)
print("              CREANDO GENERADORES")
print("=" * 70)


train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest",

    validation_split=VALIDATION_SPLIT
)


# ==========================================================
# GENERADOR DE VALIDACIÓN
# ==========================================================

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=VALIDATION_SPLIT
)


# ==========================================================
# GENERADOR DE TEST
# ==========================================================

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input
)


# ==========================================================
# TRAIN
# ==========================================================

print()
print("Creando TRAIN generator...")


train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=True,

    seed=SEED,

    subset="training"
)


# ==========================================================
# VALIDATION
# ==========================================================

print()
print("Creando VALIDATION generator...")


validation_generator = validation_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=False,

    seed=SEED,

    subset="validation"
)


# ==========================================================
# TEST
# ==========================================================

print()
print("Creando TEST generator...")


test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=False
)


# ==========================================================
# VERIFICAR MAPEO
# ==========================================================

print()
print("=" * 70)
print("              MAPEO DE CLASES")
print("=" * 70)

print(
    train_generator.class_indices
)


expected_mapping = {
    "Enojo": 0,
    "Felicidad": 1,
    "Tristeza": 2,
    "Neutral": 3
}


if train_generator.class_indices != expected_mapping:

    raise ValueError(
        "El mapeo de clases no coincide con el esperado.\n"
        f"Esperado: {expected_mapping}\n"
        f"Encontrado: "
        f"{train_generator.class_indices}"
    )


print(
    "✅ Mapeo correcto."
)


# ==========================================================
# GUARDAR MAPEO
# ==========================================================

CLASS_INDICES_PATH = os.path.join(

    MODEL_DIR,

    "class_indices.json"
)


with open(

    CLASS_INDICES_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        train_generator.class_indices,

        file,

        ensure_ascii=False,

        indent=4

    )


print(
    f"✅ Mapeo guardado en:\n"
    f"{CLASS_INDICES_PATH}"
)


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

print()
print("=" * 70)
print("              CLASS WEIGHTS")
print("=" * 70)


train_labels = train_generator.classes


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        train_labels
    ),

    y=train_labels

)


class_weights = {

    int(class_id): float(weight)

    for class_id, weight in zip(

        np.unique(
            train_labels
        ),

        class_weights_array

    )

}


for class_id, weight in class_weights.items():

    print(
        f"{CLASSES[class_id]:12s} -> "
        f"{weight:.4f}"
    )


# ==========================================================
# CONSTRUIR RESNET50
# ==========================================================

print()
print("=" * 70)
print("              CONSTRUYENDO RESNET50")
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


base_model.trainable = False


print(
    "✅ ResNet50 cargada."
)

print(
    "✅ ResNet50 congelada para Fase 1."
)


# ==========================================================
# CAPAS SUPERIORES
# ==========================================================

inputs = layers.Input(

    shape=(

        IMG_SIZE[0],

        IMG_SIZE[1],

        3

    ),

    name="imagen"

)


x = base_model(

    inputs,

    training=False

)


x = layers.GlobalAveragePooling2D()(x)


x = layers.BatchNormalization()(x)


x = layers.Dense(

    256,

    activation="relu"

)(x)


x = layers.Dropout(

    0.40

)(x)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    dtype="float32",

    name="emociones"

)(x)


model = models.Model(

    inputs=inputs,

    outputs=outputs,

    name="EmotiScan_ResNet50"

)


# ==========================================================
# COMPILAR FASE 1
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4

    ),

    loss="categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)


# ==========================================================
# RESUMEN
# ==========================================================

print()
print("=" * 70)
print("              RESUMEN DEL MODELO")
print("=" * 70)

model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

BEST_MODEL_PATH = os.path.join(

    MODEL_DIR,

    "mejor_modelo.keras"

)


FINAL_MODEL_PATH = os.path.join(

    MODEL_DIR,

    "modelo_emociones.keras"

)


CSV_LOG_PATH = os.path.join(

    RESULTS_DIR,

    "entrenamiento.csv"

)


callbacks = [

    ModelCheckpoint(

        BEST_MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True,

        verbose=1

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=3,

        min_lr=1e-7,

        verbose=1

    ),

    CSVLogger(

        CSV_LOG_PATH,

        append=False

    )

]


# ==========================================================
# FASE 1
# ==========================================================

print()
print("=" * 70)
print("              FASE 1 - TRANSFER LEARNING")
print("=" * 70)

print(
    f"Épocas: {INITIAL_EPOCHS}"
)


history_initial = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)


# ==========================================================
# CARGAR MEJOR MODELO
# ==========================================================

print()
print("=" * 70)
print("              CARGANDO MEJOR MODELO")
print("=" * 70)


model = tf.keras.models.load_model(

    BEST_MODEL_PATH

)


print(
    "✅ Mejor modelo cargado."
)


# ==========================================================
# OBTENER RESNET50
# ==========================================================

base_model = None


for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer

        break


if base_model is None:

    raise RuntimeError(
        "No se encontró la base ResNet50."
    )


print(
    "✅ ResNet50 localizada."
)


# ==========================================================
# FASE 2 - FINE TUNING
# ==========================================================

print()
print("=" * 70)
print("              FASE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


# Congelar todas las capas excepto las últimas 20

for layer in base_model.layers[:-20]:

    layer.trainable = False


# BatchNormalization congelada

for layer in base_model.layers:

    if isinstance(

        layer,

        tf.keras.layers.BatchNormalization

    ):

        layer.trainable = False


print(
    "✅ Últimas 20 capas habilitadas."
)


# ==========================================================
# COMPILAR FINE TUNING
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)


# ==========================================================
# FASE 2
# ==========================================================

history_fine = model.fit(

    train_generator,

    validation_data=validation_generator,

    initial_epoch=INITIAL_EPOCHS,

    epochs=TOTAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)


# ==========================================================
# CARGAR MEJOR MODELO FINAL
# ==========================================================

print()
print("=" * 70)
print("              CARGANDO MEJOR MODELO FINAL")
print("=" * 70)


model = tf.keras.models.load_model(

    BEST_MODEL_PATH

)


# ==========================================================
# GUARDAR MODELO KERAS
# ==========================================================

model.save(

    FINAL_MODEL_PATH

)


print(
    f"✅ Modelo guardado:\n"
    f"{FINAL_MODEL_PATH}"
)


# ==========================================================
# EVALUACIÓN SOBRE TEST
# ==========================================================

print()
print("=" * 70)
print("              EVALUACIÓN FINAL")
print("=" * 70)


test_generator.reset()


test_loss, test_accuracy = model.evaluate(

    test_generator,

    verbose=1

)


print()

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================================
# PREDICCIONES
# ==========================================================

print()
print("Generando predicciones...")


test_generator.reset()


predictions = model.predict(

    test_generator,

    verbose=1

)


y_pred = np.argmax(

    predictions,

    axis=1

)


y_true = test_generator.classes


accuracy = accuracy_score(

    y_true,

    y_pred

)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print()
print("=" * 70)
print("              CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    ),

    target_names=CLASSES,

    digits=4,

    zero_division=0

)


print()

print(report)


REPORT_PATH = os.path.join(

    RESULTS_DIR,

    "classification_report.txt"

)


with open(

    REPORT_PATH,

    "w",

    encoding="utf-8"

) as file:

    file.write(report)


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )

)


print()
print("=" * 70)
print("              MATRIZ DE CONFUSIÓN")
print("=" * 70)

print()

print(cm)


CM_NPY_PATH = os.path.join(

    RESULTS_DIR,

    "matriz_confusion.npy"

)


np.save(

    CM_NPY_PATH,

    cm

)


# ==========================================================
# MATRIZ DE CONFUSIÓN COMO IMAGEN
# ==========================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASSES

)


fig, ax = plt.subplots(

    figsize=(8, 7)

)


disp.plot(

    ax=ax,

    values_format="d",

    cmap="Blues"

)


plt.title(
    "Matriz de Confusión - EmotiScan"
)


plt.tight_layout()


CM_IMAGE_PATH = os.path.join(

    RESULTS_DIR,

    "matriz_confusion.png"

)


plt.savefig(

    CM_IMAGE_PATH,

    dpi=150,

    bbox_inches="tight"

)


plt.close()


# ==========================================================
# GRÁFICAS DE ENTRENAMIENTO
# ==========================================================

initial_acc = history_initial.history.get(
    "accuracy",
    []
)

initial_val_acc = history_initial.history.get(
    "val_accuracy",
    []
)

initial_loss = history_initial.history.get(
    "loss",
    []
)

initial_val_loss = history_initial.history.get(
    "val_loss",
    []
)


fine_acc = history_fine.history.get(
    "accuracy",
    []
)

fine_val_acc = history_fine.history.get(
    "val_accuracy",
    []
)

fine_loss = history_fine.history.get(
    "loss",
    []
)

fine_val_loss = history_fine.history.get(
    "val_loss",
    []
)


all_acc = (
    initial_acc +
    fine_acc
)

all_val_acc = (
    initial_val_acc +
    fine_val_acc
)

all_loss = (
    initial_loss +
    fine_loss
)

all_val_loss = (
    initial_val_loss +
    fine_val_loss
)


epochs_range = range(

    1,

    len(all_acc) + 1

)


# ==========================================================
# ACCURACY
# ==========================================================

plt.figure(

    figsize=(10, 6)

)


plt.plot(

    epochs_range,

    all_acc,

    label="Train Accuracy"

)


plt.plot(

    epochs_range,

    all_val_acc,

    label="Validation Accuracy"

)


plt.axvline(

    x=INITIAL_EPOCHS,

    linestyle="--",

    label="Inicio Fine Tuning"

)


plt.title(

    "EmotiScan - Accuracy"

)


plt.xlabel(

    "Época"

)


plt.ylabel(

    "Accuracy"

)


plt.legend()


plt.grid(True)


ACCURACY_PATH = os.path.join(

    RESULTS_DIR,

    "accuracy.png"

)


plt.savefig(

    ACCURACY_PATH,

    dpi=150,

    bbox_inches="tight"

)


plt.close()


# ==========================================================
# LOSS
# ==========================================================

plt.figure(

    figsize=(10, 6)

)


plt.plot(

    epochs_range,

    all_loss,

    label="Train Loss"

)


plt.plot(

    epochs_range,

    all_val_loss,

    label="Validation Loss"

)


plt.axvline(

    x=INITIAL_EPOCHS,

    linestyle="--",

    label="Inicio Fine Tuning"

)


plt.title(

    "EmotiScan - Loss"

)


plt.xlabel(

    "Época"

)


plt.ylabel(

    "Loss"

)


plt.legend()


plt.grid(True)


LOSS_PATH = os.path.join(

    RESULTS_DIR,

    "loss.png"

)


plt.savefig(

    LOSS_PATH,

    dpi=150,

    bbox_inches="tight"

)


plt.close()


# ==========================================================
# MÉTRICAS JSON
# ==========================================================

metrics = {

    "image_size": "224x224",

    "input_shape": [
        224,
        224,
        3
    ],

    "classes": CLASSES,

    "class_indices":
        train_generator.class_indices,

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "accuracy":
        float(accuracy),

    "confusion_matrix":
        cm.tolist()

}


METRICS_PATH = os.path.join(

    RESULTS_DIR,

    "metricas.json"

)


with open(

    METRICS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metrics,

        file,

        ensure_ascii=False,

        indent=4

    )


# ==========================================================
# CONVERTIR A TFLITE
# ==========================================================

print()
print("=" * 70)
print("              CONVIRTIENDO A TFLITE")
print("=" * 70)


TFLITE_PATH = os.path.join(

    MODEL_DIR,

    "modelo_resnet50_emociones.tflite"

)


converter = tf.lite.TFLiteConverter.from_keras_model(

    model

)


# Optimizaciones compatibles con producción

converter.optimizations = [

    tf.lite.Optimize.DEFAULT

]


tflite_model = converter.convert()


with open(

    TFLITE_PATH,

    "wb"

) as file:

    file.write(

        tflite_model

    )


print(
    f"✅ TFLite generado:\n"
    f"{TFLITE_PATH}"
)


# ==========================================================
# VERIFICAR TFLITE
# ==========================================================

print()
print("=" * 70)
print("              VERIFICANDO TFLITE")
print("=" * 70)


interpreter = tf.lite.Interpreter(

    model_path=TFLITE_PATH

)


interpreter.allocate_tensors()


input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()


print()

print(
    "Input:"
)

print(
    input_details[0]["shape"]
)

print(
    input_details[0]["dtype"]
)


print()

print(
    "Output:"
)

print(
    output_details[0]["shape"]
)

print(
    output_details[0]["dtype"]
)


expected_input_shape = np.array(

    [1, 224, 224, 3]

)


if not np.array_equal(

    input_details[0]["shape"],

    expected_input_shape

):

    print(
        "⚠️ ADVERTENCIA: "
        "el tamaño de entrada no coincide."
    )

else:

    print(
        "✅ Entrada TFLite correcta: "
        "[1, 224, 224, 3]"
    )


# ==========================================================
# TAMAÑO DE ARCHIVOS
# ==========================================================

print()
print("=" * 70)
print("              ARCHIVOS GENERADOS")
print("=" * 70)


generated_files = [

    BEST_MODEL_PATH,

    FINAL_MODEL_PATH,

    CLASS_INDICES_PATH,

    TFLITE_PATH,

    CSV_LOG_PATH,

    REPORT_PATH,

    CM_NPY_PATH,

    CM_IMAGE_PATH,

    ACCURACY_PATH,

    LOSS_PATH,

    METRICS_PATH

]


for filepath in generated_files:

    if os.path.exists(filepath):

        size_mb = (

            os.path.getsize(filepath)

            /

            (1024 * 1024)

        )

        print(

            f"✅ {filepath} "
            f"({size_mb:.2f} MB)"

        )

    else:

        print(

            f"❌ NO GENERADO: "
            f"{filepath}"

        )


# ==========================================================
# RESULTADO FINAL
# ==========================================================

print()
print("=" * 70)
print("              ENTRENAMIENTO FINALIZADO")
print("=" * 70)

print()

print(
    f"Accuracy TEST: "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    "Modelos:"
)

print(
    "✅ modelo/mejor_modelo.keras"
)

print(
    "✅ modelo/modelo_emociones.keras"
)

print(
    "✅ modelo/modelo_resnet50_emociones.tflite"
)

print()

print(
    "Resultados:"
)

print(
    "✅ resultados/classification_report.txt"
)

print(
    "✅ resultados/matriz_confusion.png"
)

print(
    "✅ resultados/accuracy.png"
)

print(
    "✅ resultados/loss.png"
)

print(
    "✅ resultados/metricas.json"
)

print()

print(
    "🎉 EmotiScan terminó el entrenamiento."
)