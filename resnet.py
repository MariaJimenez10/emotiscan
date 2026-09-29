import os
import json
import logging
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
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
# CONFIGURACIÓN
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================================
# CLASES
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

# IMPORTANTE:
# Todo el entrenamiento utiliza imágenes de 224 x 224.

IMG_SIZE = (224, 224)

# Reducido para disminuir consumo de RAM.
BATCH_SIZE = 16

VALIDATION_SPLIT = 0.20

SEED = 42

# Primera fase
INITIAL_EPOCHS = 15

# Segunda fase
FINE_TUNE_EPOCHS = 15

TOTAL_EPOCHS = (
    INITIAL_EPOCHS +
    FINE_TUNE_EPOCHS
)


# ==========================================================
# SEMILLAS
# ==========================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# COMPROBAR DATASET
# ==========================================================

logger.info("=" * 70)
logger.info("VERIFICANDO DATASET")
logger.info("=" * 70)

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta dataset:\n{DATASET_DIR}"
    )

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta train:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"No existe la carpeta test:\n{TEST_DIR}"
    )


# ==========================================================
# COMPROBAR CARPETAS DE CLASES
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

    if not os.path.exists(train_class_dir):

        raise FileNotFoundError(
            f"Falta la carpeta:\n{train_class_dir}"
        )

    if not os.path.exists(test_class_dir):

        raise FileNotFoundError(
            f"Falta la carpeta:\n{test_class_dir}"
        )


logger.info(
    "Dataset encontrado correctamente."
)


# ==========================================================
# CONTAR IMÁGENES
# ==========================================================

def count_images(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    total = 0

    if not os.path.exists(folder):
        return 0

    for filename in os.listdir(folder):

        filepath = os.path.join(
            folder,
            filename
        )

        if os.path.isfile(filepath):

            if filename.lower().endswith(
                extensions
            ):

                total += 1

    return total


# ==========================================================
# MOSTRAR CANTIDAD DE IMÁGENES
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("IMÁGENES DEL DATASET")
logger.info("=" * 70)

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

    logger.info(
        f"{class_name:12s} | "
        f"Train: {train_count:5d} | "
        f"Test: {test_count:5d}"
    )


logger.info("-" * 70)

logger.info(
    f"TOTAL TRAIN: {total_train}"
)

logger.info(
    f"TOTAL TEST : {total_test}"
)

logger.info(
    f"TOTAL DATASET: {total_train + total_test}"
)


# ==========================================================
# DATA GENERATORS
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("CREANDO GENERADORES")
logger.info("=" * 70)


# ==========================================================
# TRAIN
# ==========================================================

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
# VALIDATION
# ==========================================================

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=VALIDATION_SPLIT
)


# ==========================================================
# TEST
# ==========================================================

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input
)


# ==========================================================
# TRAIN GENERATOR
# ==========================================================

logger.info("")
logger.info("Creando TRAIN generator...")

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    # IMPORTANTE:
    # Keras utilizará 224 x 224.
    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=True,

    seed=SEED,

    subset="training"
)


# ==========================================================
# VALIDATION GENERATOR
# ==========================================================

logger.info("")
logger.info("Creando VALIDATION generator...")

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
# TEST GENERATOR
# ==========================================================

logger.info("")
logger.info("Creando TEST generator...")

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    classes=CLASSES,

    shuffle=False
)


# ==========================================================
# VERIFICAR TAMAÑO DE LAS IMÁGENES
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("VERIFICANDO TAMAÑO DE ENTRADA")
logger.info("=" * 70)

logger.info(
    f"Tamaño esperado: {IMG_SIZE[0]} x {IMG_SIZE[1]} píxeles"
)

logger.info(
    f"Forma de entrada del generador: "
    f"{train_generator.image_shape}"
)


if train_generator.image_shape != (
    IMG_SIZE[0],
    IMG_SIZE[1],
    3
):

    raise ValueError(
        "El tamaño de entrada no coincide con 224x224x3."
    )


logger.info(
    "Tamaño de entrada correcto: 224 x 224 x 3"
)


# ==========================================================
# MAPEO DE CLASES
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("MAPEO DE CLASES")
logger.info("=" * 70)

logger.info(
    str(train_generator.class_indices)
)


# ==========================================================
# GUARDAR MAPEO
# ==========================================================

class_indices_path = os.path.join(
    MODEL_DIR,
    "class_indices.json"
)

with open(
    class_indices_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        train_generator.class_indices,
        file,
        ensure_ascii=False,
        indent=4
    )


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("CALCULANDO CLASS WEIGHTS")
logger.info("=" * 70)

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

    for class_id, weight
    in zip(
        np.unique(train_labels),
        class_weights_array
    )
}


for class_id, weight in class_weights.items():

    logger.info(
        f"{CLASSES[class_id]:12s} -> {weight:.4f}"
    )


# ==========================================================
# CONSTRUIR RESNET50
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("CONSTRUYENDO RESNET50")
logger.info("=" * 70)

logger.info(
    "Entrada: 224 x 224 x 3"
)


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
# CONGELAR RESNET50
# ==========================================================

base_model.trainable = False


logger.info(
    "ResNet50 congelada para la primera fase."
)


# ==========================================================
# MODELO FINAL
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


x = layers.Dropout(
    0.40
)(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.BatchNormalization()(x)


x = layers.Dropout(
    0.30
)(x)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    name="emociones"
)(x)


model = models.Model(

    inputs=inputs,

    outputs=outputs,

    name="EmotiScan_ResNet50"
)


# ==========================================================
# COMPILACIÓN INICIAL
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
# MOSTRAR MODELO
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("RESUMEN DEL MODELO")
logger.info("=" * 70)

model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

MODEL_PATH = os.path.join(

    MODEL_DIR,

    "modelo_emociones.keras"
)


CHECKPOINT_PATH = os.path.join(

    MODEL_DIR,

    "mejor_modelo.keras"
)


CSV_LOG_PATH = os.path.join(

    RESULTS_DIR,

    "entrenamiento.csv"
)


callbacks = [

    ModelCheckpoint(

        CHECKPOINT_PATH,

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
# TRANSFER LEARNING
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("FASE 1 - TRANSFER LEARNING")
logger.info("=" * 70)

logger.info(
    f"Épocas: {INITIAL_EPOCHS}"
)

logger.info(
    "ResNet50 está congelada."
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
# FASE 2
# FINE TUNING
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("FASE 2 - FINE TUNING")
logger.info("=" * 70)


# Activar ResNet50
base_model.trainable = True


# ==========================================================
# CONGELAR PRIMERAS CAPAS
# ==========================================================

FINE_TUNE_FROM = 140


for layer in base_model.layers[
    :FINE_TUNE_FROM
]:

    layer.trainable = False


# ==========================================================
# CONGELAR BATCH NORMALIZATION
# ==========================================================

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization
    ):

        layer.trainable = False


logger.info(
    f"Fine tuning desde la capa {FINE_TUNE_FROM}."
)


# ==========================================================
# RECOMPILAR
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
# FINE TUNING
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
# GUARDAR MODELO FINAL
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("GUARDANDO MODELO FINAL")
logger.info("=" * 70)


model.save(
    MODEL_PATH
)


logger.info(
    f"Modelo guardado en:\n{MODEL_PATH}"
)


# ==========================================================
# EVALUACIÓN TEST
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("EVALUACIÓN FINAL SOBRE TEST")
logger.info("=" * 70)


test_generator.reset()


test_loss, test_accuracy = model.evaluate(

    test_generator,

    verbose=1
)


logger.info("")
logger.info(
    f"TEST LOSS     : {test_loss:.4f}"
)

logger.info(
    f"TEST ACCURACY : {test_accuracy:.4f}"
)


# ==========================================================
# PREDICCIONES
# ==========================================================

logger.info("")
logger.info(
    "Generando predicciones..."
)


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


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(

    y_true,

    y_pred
)


logger.info("")
logger.info(
    f"Accuracy final: {accuracy:.4f}"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("CLASSIFICATION REPORT")
logger.info("=" * 70)


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


print("")
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

    file.write(
        report
    )


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("MATRIZ DE CONFUSIÓN")
logger.info("=" * 70)


cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )
)


print("")

print(
    " " * 18 +
    " ".join(
        f"{name:>12}"
        for name in CLASSES
    )
)


for i, row in enumerate(cm):

    print(

        f"{CLASSES[i]:>12}       " +

        " ".join(
            f"{value:>12}"
            for value in row
        )
    )


CM_PATH = os.path.join(

    RESULTS_DIR,

    "matriz_confusion.npy"
)


np.save(

    CM_PATH,

    cm
)


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

    "test_loss": float(
        test_loss
    ),

    "test_accuracy": float(
        test_accuracy
    ),

    "accuracy": float(
        accuracy
    ),

    "confusion_matrix": cm.tolist()
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
# GRÁFICAS
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("GENERANDO GRÁFICAS")
logger.info("=" * 70)


import matplotlib.pyplot as plt


# ==========================================================
# HISTORIES
# ==========================================================

initial_acc = history_initial.history[
    "accuracy"
]

initial_val_acc = history_initial.history[
    "val_accuracy"
]

initial_loss = history_initial.history[
    "loss"
]

initial_val_loss = history_initial.history[
    "val_loss"
]


fine_acc = history_fine.history[
    "accuracy"
]

fine_val_acc = history_fine.history[
    "val_accuracy"
]

fine_loss = history_fine.history[
    "loss"
]

fine_val_loss = history_fine.history[
    "val_loss"
]


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

plt.grid(
    True
)


accuracy_plot = os.path.join(

    RESULTS_DIR,

    "accuracy.png"
)


plt.savefig(

    accuracy_plot,

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

plt.grid(
    True
)


loss_plot = os.path.join(

    RESULTS_DIR,

    "loss.png"
)


plt.savefig(

    loss_plot,

    dpi=150,

    bbox_inches="tight"
)


plt.close()


# ==========================================================
# MATRIZ DE CONFUSIÓN COMO IMAGEN
# ==========================================================

plt.figure(

    figsize=(8, 7)
)


plt.imshow(cm)


plt.title(
    "Matriz de Confusión - EmotiScan"
)


plt.colorbar()


plt.xticks(

    np.arange(NUM_CLASSES),

    CLASSES,

    rotation=45,

    ha="right"
)


plt.yticks(

    np.arange(NUM_CLASSES),

    CLASSES
)


threshold = cm.max() / 2.0


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.xlabel(
    "Predicción"
)

plt.ylabel(
    "Real"
)


plt.tight_layout()


cm_plot = os.path.join(

    RESULTS_DIR,

    "matriz_confusion.png"
)


plt.savefig(

    cm_plot,

    dpi=150,

    bbox_inches="tight"
)


plt.close()


# ==========================================================
# FINAL
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("ENTRENAMIENTO FINALIZADO")
logger.info("=" * 70)


logger.info(
    f"Modelo final:\n{MODEL_PATH}"
)


logger.info(
    f"Mejor modelo:\n{CHECKPOINT_PATH}"
)


logger.info(
    f"Resultados:\n{RESULTS_DIR}"
)


logger.info("")
logger.info("Archivos generados:")

logger.info(
    "  - modelo/modelo_emociones.keras"
)

logger.info(
    "  - modelo/mejor_modelo.keras"
)

logger.info(
    "  - modelo/class_indices.json"
)

logger.info(
    "  - resultados/entrenamiento.csv"
)

logger.info(
    "  - resultados/classification_report.txt"
)

logger.info(
    "  - resultados/metricas.json"
)

logger.info(
    "  - resultados/matriz_confusion.npy"
)

logger.info(
    "  - resultados/matriz_confusion.png"
)

logger.info(
    "  - resultados/accuracy.png"
)

logger.info(
    "  - resultados/loss.png"
)
