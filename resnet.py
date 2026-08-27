import os
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

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
    )


# ==========================================================
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
)


# ==========================================================
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

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)


# ==========================================================
# PREPROCESAR TEST
# ==========================================================

X_test_processed = preprocess_input(
    X_test.copy()
)


# ==========================================================
# CLASS WEIGHTS
# ==========================================================

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        y_train
    ),

    y=y_train
)


class_weights = {
    i: float(peso)
    for i, peso in enumerate(
        class_weights_array
    )
}


print("\n======================================")
print("       CLASS WEIGHTS")
print("======================================")

for i, emocion in enumerate(EMOCIONES):

    print(
        f"{i} - {emocion}: "
        f"{class_weights[i]:.4f}"
    )


# ==========================================================
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
        3
    )
)

base_model._name = "resnet50_base"


# ==========================================================
# CONGELAR RESNET50
# ==========================================================

base_model.trainable = False


# ==========================================================
# CAPAS SUPERIORES
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

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

)


# ==========================================================
# COMPILAR
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==========================================================
# RESUMEN
# ==========================================================

model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    EarlyStopping(

        monitor="val_accuracy",

        patience=3,

        restore_best_weights=True,

        verbose=1

    ),

    ModelCheckpoint(

        "mejor_modelo.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1

    ),

    ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1

    )

]


# ==========================================================
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

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

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
)


# ==========================================================
# FINE TUNING
# ==========================================================

print("\n======================================")
print("             FINE TUNING")
print("======================================\n")


base_model.trainable = True


# Congelar todas excepto últimas 20
for layer in base_model.layers[:-20]:

    layer.trainable = False


# BatchNormalization siempre congelada
for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


print(
    "✅ Últimas 20 capas habilitadas"
)


# ==========================================================
# COMPILAR FINE TUNING
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ==========================================================
# FINE TUNING
# ==========================================================

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

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)


# ==========================================================
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
)


# ==========================================================
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

)


# ==========================================================
# PREDICCIONES
# ==========================================================

y_pred = np.argmax(

    predicciones,

    axis=1

)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n======================================")
print("              ACCURACY")
print("======================================")


print(
    f"{accuracy * 100:.2f}%"
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

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


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

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

)


plt.title(
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


# ==========================================================
# FINAL
# ==========================================================

print("\n======================================")
print("      PROCESO TERMINADO")
print("======================================")


print("\n🎉 Entrenamiento completado correctamente.")

print("\nModelos disponibles:")

print("✅ mejor_modelo.keras")
print("✅ modelo_resnet50_emociones.keras")
print("✅ modelo_resnet50_emociones.h5")
print("✅ modelo_resnet50_emociones.tflite")