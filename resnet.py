import os
<<<<<<< HEAD
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
=======
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

<<<<<<< HEAD
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
=======
IMG_SIZE = 224
BATCH_SIZE = 16

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

NUM_CLASES = len(EMOCIONES)

np.random.seed(42)
tf.random.set_seed(42)


# ==========================================================
# GPU / MIXED PRECISION
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

print("\n======================================")
print("       CONFIGURACIÓN DEL ENTORNO")
print("======================================")

if gpus:
    print("✅ GPU DETECTADA")
    print(gpus)

    mixed_precision.set_global_policy("mixed_float16")
else:
    print("⚠️ NO SE DETECTÓ GPU")
    print("Se utilizará CPU.")


# ==========================================================
# INFORMACIÓN
# ==========================================================

print("\n======================================")
print("       ENTRENAMIENTO RESNET50")
print("======================================")

print("\nClases:")

for i, emocion in enumerate(EMOCIONES):
    print(f"{i} -> {emocion}")


# ==========================================================
# VERIFICAR DATASETS
# ==========================================================

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"No se encontró: {TRAIN_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"No se encontró: {TEST_PATH}"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    )


# ==========================================================
<<<<<<< HEAD
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
=======
# FUNCIÓN PARA CARGAR IMÁGENES
# ==========================================================

def cargar_dataset(dataset_path, nombre):

    X = []
    y = []

    print("\n======================================")
    print(f"CARGANDO {nombre}")
    print("======================================")

    extensiones_validas = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    for idx, emocion in enumerate(EMOCIONES):

        carpeta = os.path.join(
            dataset_path,
            emocion
        )

        if not os.path.exists(carpeta):
            print(f"❌ No existe: {carpeta}")
            continue

        archivos = [
            archivo
            for archivo in os.listdir(carpeta)
            if archivo.lower().endswith(extensiones_validas)
        ]

        contador = 0

        for archivo in archivos:

            ruta = os.path.join(
                carpeta,
                archivo
            )

            img = cv2.imread(ruta)

            if img is None:
                print(
                    f"⚠️ No se pudo leer: {ruta}"
                )
                continue

            # BGR -> RGB
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            # Redimensionar
            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            X.append(img)
            y.append(idx)

            contador += 1

        print(
            f"{idx} - {emocion}: "
            f"{contador} imágenes"
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    print(
        f"\nTOTAL {nombre}: {len(X)}"
    )

    return X, y


# ==========================================================
# CARGAR TRAIN
# ==========================================================

X_train, y_train = cargar_dataset(
    TRAIN_PATH,
    "TRAIN"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
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
=======
# CARGAR TEST
# ==========================================================

X_test, y_test = cargar_dataset(
    TEST_PATH,
    "TEST"
)


# ==========================================================
# VERIFICAR
# ==========================================================

if len(X_train) == 0:
    raise ValueError(
        "No hay imágenes en TRAIN."
    )

if len(X_test) == 0:
    raise ValueError(
        "No hay imágenes en TEST."
    )


# ==========================================================
# DISTRIBUCIÓN
# ==========================================================

print("\n======================================")
print("       DISTRIBUCIÓN TRAIN")
print("======================================")

for i, emocion in enumerate(EMOCIONES):

    cantidad = np.sum(
        y_train == i
    )

    print(
        f"{i} - {emocion}: {cantidad}"
    )


print("\n======================================")
print("       DISTRIBUCIÓN TEST")
print("======================================")

for i, emocion in enumerate(EMOCIONES):

    cantidad = np.sum(
        y_test == i
    )

    print(
        f"{i} - {emocion}: {cantidad}"
    )


# ==========================================================
# DATA AUGMENTATION
# ==========================================================

datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

<<<<<<< HEAD
    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest",

    validation_split=VALIDATION_SPLIT
=======
    horizontal_flip=True,

    fill_mode="nearest"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
# VALIDATION
# ==========================================================

validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=VALIDATION_SPLIT
=======
# PREPROCESAR TEST
# ==========================================================

X_test_processed = preprocess_input(
    X_test.copy()
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
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


=======
# CLASS WEIGHTS
# ==========================================================

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
<<<<<<< HEAD
        train_labels
    ),

    y=train_labels
=======
        y_train
    ),

    y=y_train
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


class_weights = {
<<<<<<< HEAD

    int(class_id): float(weight)

    for class_id, weight
    in zip(
        np.unique(train_labels),
=======
    i: float(peso)
    for i, peso in enumerate(
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        class_weights_array
    )
}


<<<<<<< HEAD
for class_id, weight in class_weights.items():

    logger.info(
        f"{CLASSES[class_id]:12s} -> {weight:.4f}"
=======
print("\n======================================")
print("       CLASS WEIGHTS")
print("======================================")

for i, emocion in enumerate(EMOCIONES):

    print(
        f"{i} - {emocion}: "
        f"{class_weights[i]:.4f}"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    )


# ==========================================================
<<<<<<< HEAD
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
=======
# CREAR RESNET50
# ==========================================================

print("\n======================================")
print("       CREANDO RESNET50")
print("======================================")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
        3
    )
)

<<<<<<< HEAD
=======
base_model._name = "resnet50_base"

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

# ==========================================================
# CONGELAR RESNET50
# ==========================================================

base_model.trainable = False


<<<<<<< HEAD
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
=======
# ==========================================================
# CAPAS SUPERIORES
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    256,
    activation="relu"
)(x)

<<<<<<< HEAD

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
=======
x = Dropout(
    0.5
)(x)


output = Dense(

    NUM_CLASES,

    activation="softmax",

    dtype="float32"

)(x)


model = Model(

    inputs=base_model.input,

    outputs=output

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
# COMPILACIÓN INICIAL
=======
# COMPILAR
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
<<<<<<< HEAD

        learning_rate=1e-4
    ),

    loss="categorical_crossentropy",
=======
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    metrics=[
        "accuracy"
    ]
<<<<<<< HEAD
=======

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
# MOSTRAR MODELO
# ==========================================================

logger.info("")
logger.info("=" * 70)
logger.info("RESUMEN DEL MODELO")
logger.info("=" * 70)

=======
# RESUMEN
# ==========================================================

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

<<<<<<< HEAD
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
=======
callbacks = [

    EarlyStopping(

        monitor="val_accuracy",

        patience=3,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        restore_best_weights=True,

        verbose=1
<<<<<<< HEAD
=======

    ),

    ModelCheckpoint(

        "mejor_modelo.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    ),

    ReduceLROnPlateau(

        monitor="val_loss",

<<<<<<< HEAD
        factor=0.3,

        patience=3,
=======
        factor=0.2,

        patience=2,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

        min_lr=1e-7,

        verbose=1
<<<<<<< HEAD
    ),

    CSVLogger(

        CSV_LOG_PATH,

        append=False
    )
=======

    )

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
]


# ==========================================================
<<<<<<< HEAD
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
=======
# ENTRENAMIENTO INICIAL
# ==========================================================

print("\n======================================")
print("       ENTRENAMIENTO INICIAL")
print("======================================\n")


history = model.fit(

    datagen.flow(

        X_train,

        y_train,

        batch_size=BATCH_SIZE,

        shuffle=True

    ),

    validation_data=(

        X_test_processed,

        y_test

    ),

    epochs=8,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
<<<<<<< HEAD
=======

)


# ==========================================================
# CARGAR MEJOR MODELO
# ==========================================================

print("\n======================================")
print("       CARGANDO MEJOR MODELO")
print("======================================")


model = load_model(
    "mejor_modelo.keras"
)

base_model = model.get_layer("resnet50")


print(
    "✅ Mejor modelo cargado"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
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
=======
# FINE TUNING
# ==========================================================

print("\n======================================")
print("             FINE TUNING")
print("======================================\n")


base_model.trainable = True


# Congelar todas excepto últimas 20
for layer in base_model.layers[:-20]:
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    layer.trainable = False


<<<<<<< HEAD
# ==========================================================
# CONGELAR BATCH NORMALIZATION
# ==========================================================

=======
# BatchNormalization siempre congelada
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
for layer in base_model.layers:

    if isinstance(
        layer,
<<<<<<< HEAD
        layers.BatchNormalization
=======
        tf.keras.layers.BatchNormalization
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
    ):

        layer.trainable = False


<<<<<<< HEAD
logger.info(
    f"Fine tuning desde la capa {FINE_TUNE_FROM}."
=======
print(
    "✅ Últimas 20 capas habilitadas"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
# RECOMPILAR
=======
# COMPILAR FINE TUNING
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5
<<<<<<< HEAD
    ),

    loss="categorical_crossentropy",
=======

    ),

    loss="sparse_categorical_crossentropy",
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    metrics=[
        "accuracy"
    ]
<<<<<<< HEAD
=======

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
# FINE TUNING
# ==========================================================

<<<<<<< HEAD
history_fine = model.fit(

    train_generator,

    validation_data=validation_generator,

    initial_epoch=INITIAL_EPOCHS,

    epochs=TOTAL_EPOCHS,
=======
history2 = model.fit(

    datagen.flow(

        X_train,

        y_train,

        batch_size=BATCH_SIZE,

        shuffle=True

    ),

    validation_data=(

        X_test_processed,

        y_test

    ),

    epochs=5,
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1
<<<<<<< HEAD
=======

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
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
=======
# CARGAR MEJOR MODELO FINAL
# ==========================================================

print("\n======================================")
print("       CARGANDO MEJOR MODELO FINAL")
print("======================================")


model = load_model(
    "mejor_modelo.keras"
)


print(
    "✅ Mejor modelo final cargado"
)


# ==========================================================
# GUARDAR KERAS
# ==========================================================

model.save(
    "modelo_resnet50_emociones.keras"
)

print(
    "✅ modelo_resnet50_emociones.keras"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
<<<<<<< HEAD
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
=======
# GUARDAR H5
# ==========================================================

model.save(
    "modelo_resnet50_emociones.h5"
)

print(
    "✅ modelo_resnet50_emociones.h5"
)


# ==========================================================
# CONVERTIR A TFLITE
# ==========================================================

print("\n======================================")
print("       CONVIRTIENDO A TFLITE")
print("======================================")


converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)


converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


tflite_model = converter.convert()


with open(
    "modelo_resnet50_emociones.tflite",
    "wb"
) as archivo:

    archivo.write(
        tflite_model
    )


print(
    "✅ modelo_resnet50_emociones.tflite"
)


# ==========================================================
# EVALUACIÓN
# ==========================================================

print("\n======================================")
print("       EVALUANDO MODELO")
print("======================================")


predicciones = model.predict(

    X_test_processed,

    batch_size=BATCH_SIZE,

    verbose=1

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
# PREDICCIONES
# ==========================================================

<<<<<<< HEAD
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


=======
y_pred = np.argmax(

    predicciones,

    axis=1

)


>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(

<<<<<<< HEAD
    y_true,

    y_pred
)


logger.info("")
logger.info(
    f"Accuracy final: {accuracy:.4f}"
=======
    y_test,

    y_pred

)


print("\n======================================")
print("              ACCURACY")
print("======================================")


print(
    f"{accuracy * 100:.2f}%"
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

<<<<<<< HEAD
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


=======
print("\n======================================")
print("       CLASSIFICATION REPORT")
print("======================================")


reporte = classification_report(

    y_test,

    y_pred,

    target_names=EMOCIONES,

    digits=4

)


print(
    reporte
)


>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

<<<<<<< HEAD
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
=======
cm = confusion_matrix(

    y_test,

    y_pred

)


print("\n======================================")
print("       MATRIZ DE CONFUSIÓN")
print("======================================")


print(cm)


# ==========================================================
# MOSTRAR MATRIZ
# ==========================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=EMOCIONES

)


disp.plot(

    cmap="Blues",

    values_format="d"

>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
)


plt.title(
<<<<<<< HEAD
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
=======
    "Matriz de Confusión - ResNet50"
)

plt.tight_layout()

plt.show()


# ==========================================================
# RESULTADO TRISTEZA
# ==========================================================

print("\n======================================")
print("       RESULTADO DE TRISTEZA")
print("======================================")


report_dict = classification_report(

    y_test,

    y_pred,

    target_names=EMOCIONES,

    output_dict=True

)


tristeza = report_dict[
    "Tristeza"
]


print(
    f"Precision: "
    f"{tristeza['precision'] * 100:.2f}%"
)


print(
    f"Recall:    "
    f"{tristeza['recall'] * 100:.2f}%"
)


print(
    f"F1-score:  "
    f"{tristeza['f1-score'] * 100:.2f}%"
)


# ==========================================================
# TAMAÑO DE ARCHIVOS
# ==========================================================

print("\n======================================")
print("       ARCHIVOS GENERADOS")
print("======================================")


archivos_generados = [

    "mejor_modelo.keras",

    "modelo_resnet50_emociones.keras",

    "modelo_resnet50_emociones.h5",

    "modelo_resnet50_emociones.tflite"

]


for archivo in archivos_generados:

    if os.path.exists(archivo):

        tamaño_mb = (

            os.path.getsize(archivo)

            / (1024 * 1024)

        )

        print(

            f"✅ {archivo} "

            f"({tamaño_mb:.2f} MB)"

        )

    else:

        print(

            f"❌ NO generado: {archivo}"

        )
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8


# ==========================================================
# FINAL
# ==========================================================

<<<<<<< HEAD
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
=======
print("\n======================================")
print("      PROCESO TERMINADO")
print("======================================")


print("\n🎉 Entrenamiento completado correctamente.")

print("\nModelos disponibles:")

print("✅ mejor_modelo.keras")
print("✅ modelo_resnet50_emociones.keras")
print("✅ modelo_resnet50_emociones.h5")
print("✅ modelo_resnet50_emociones.tflite")
>>>>>>> b2b5a0ad109606f33b3ce92679f1ab8de8c621f8
