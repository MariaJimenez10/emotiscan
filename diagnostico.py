
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. RUTAS Y CONFIGURACIÓN
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "modelo", "resnet50_emociones_mejorado.keras"
)

TFLITE_PATH = os.path.join(
    BASE_DIR, "modelo", "modelo_resnet50_emociones.tflite"
)

CLASSES_PATH = os.path.join(
    BASE_DIR, "modelo", "clases.json"
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "test"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASSES_ESPERADAS = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# =========================================================
# 2. VERIFICAR ARCHIVOS
# =========================================================

for ruta in [MODEL_PATH, TFLITE_PATH, CLASSES_PATH, TEST_DIR]:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo o directorio:\n{ruta}"
        )

with open(CLASSES_PATH, "r", encoding="utf-8") as archivo:
    classes = json.load(archivo)

print("\nCLASES GUARDADAS:", classes)

if classes != CLASSES_ESPERADAS:
    raise ValueError(
        "El orden de clases no coincide.\n"
        f"Esperado: {CLASSES_ESPERADAS}\n"
        f"Encontrado: {classes}"
    )


# =========================================================
# 3. CARGAR MODELO KERAS
# =========================================================

print("\nCargando modelo Keras...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("ENTRADA KERAS:", model.input_shape)
print("SALIDA KERAS:", model.output_shape)

if model.output_shape[-1] != len(classes):
    raise ValueError(
        "El modelo Keras no tiene cuatro salidas como se esperaba."
    )


# =========================================================
# 4. CARGAR DATASET DE PRUEBA
# =========================================================

print("\nCargando imágenes de prueba...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test = datagen.flow_from_directory(
    TEST_DIR,
    classes=classes,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

if test.samples == 0:
    raise ValueError(
        "No se encontraron imágenes en dataset/test."
    )

y_true = test.classes.copy()

print("Total de imágenes:", len(y_true))
print("Orden de clases:", test.class_indices)


# =========================================================
# 5. EVALUAR MODELO KERAS
# =========================================================

print("\nEvaluando modelo Keras...")

test.reset()

keras_probs = model.predict(
    test,
    verbose=1
)

if keras_probs.shape[1] != len(classes):
    raise ValueError(
        f"Keras produjo {keras_probs.shape[1]} salidas; "
        f"se esperaban {len(classes)}."
    )

keras_pred = np.argmax(keras_probs, axis=1)

print("\n========== RESULTADO KERAS ==========")

print(
    classification_report(
        y_true,
        keras_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

print("MATRIZ DE CONFUSIÓN (KERAS):")
print(
    confusion_matrix(
        y_true,
        keras_pred,
        labels=list(range(len(classes)))
    )
)


# =========================================================
# 6. CARGAR MODELO TFLITE
# =========================================================

print("\nCargando modelo TensorFlow Lite...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_PATH
)

interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("ENTRADA TFLITE:", inp["shape"], inp["dtype"])
print("SALIDA TFLITE:", out["shape"], out["dtype"])

if int(inp["shape"][-1]) != 3:
    raise ValueError(
        "El modelo TFLite no espera imágenes de tres canales."
    )

if int(out["shape"][-1]) != len(classes):
    raise ValueError(
        "El modelo TFLite no tiene cuatro salidas como se esperaba."
    )


# =========================================================
# 7. EVALUAR TFLITE IMAGEN POR IMAGEN
# =========================================================

print("\nEvaluando modelo TFLite...")

tflite_preds = []
tflite_probs = []

test.reset()

for lote_numero in range(len(test)):
    x, _ = next(test)

    for imagen in x:

        # TFLite espera una sola imagen por llamada.
        imagen = np.expand_dims(imagen, axis=0)

        # Convertir a la forma y tipo de entrada esperados.
        if np.issubdtype(inp["dtype"], np.integer):
            escala, cero = inp["quantization"]

            if escala <= 0:
                raise ValueError(
                    "La entrada TFLite está cuantizada, "
                    "pero no tiene una escala válida."
                )

            imagen = np.round(imagen / escala + cero)

            limites = np.iinfo(inp["dtype"])
            imagen = np.clip(
                imagen,
                limites.min,
                limites.max
            )

        imagen = imagen.astype(inp["dtype"])

        interpreter.set_tensor(
            inp["index"],
            imagen
        )

        interpreter.invoke()

        resultado = interpreter.get_tensor(
            out["index"]
        )[0]

        # Convertir salida cuantizada a valores reales.
        if np.issubdtype(out["dtype"], np.integer):
            escala_salida, cero_salida = out["quantization"]

            if escala_salida > 0:
                resultado = (
                    resultado.astype(np.float32) - cero_salida
                ) * escala_salida
            else:
                raise ValueError(
                    "La salida TFLite está cuantizada, "
                    "pero no tiene una escala válida."
                )

        resultado = np.asarray(
            resultado,
            dtype=np.float32
        )

        tflite_probs.append(resultado)
        tflite_preds.append(int(np.argmax(resultado)))

    if (lote_numero + 1) % 20 == 0 or lote_numero + 1 == len(test):
        print(
            f"Lotes procesados: {lote_numero + 1}/{len(test)}"
        )

tflite_preds = np.asarray(tflite_preds)
tflite_probs = np.asarray(tflite_probs)

if len(tflite_preds) != len(y_true):
    raise ValueError(
        f"Cantidad de predicciones TFLite: {len(tflite_preds)}; "
        f"cantidad de etiquetas reales: {len(y_true)}."
    )


# =========================================================
# 8. MOSTRAR RESULTADOS TFLITE
# =========================================================

print("\n========== RESULTADO TFLITE ==========")

print(
    classification_report(
        y_true,
        tflite_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

print("MATRIZ DE CONFUSIÓN (TFLITE):")

print(
    confusion_matrix(
        y_true,
        tflite_preds,
        labels=list(range(len(classes)))
    )
)


# =========================================================
# 9. COMPARAR KERAS CONTRA TFLITE
# =========================================================

porcentaje_iguales = np.mean(
    keras_pred == tflite_preds
) * 100

print("\n========== COMPARACIÓN ==========")

print(
    "Predicciones Keras y TFLite iguales:",
    f"{porcentaje_iguales:.2f}%"
)

print("\n========== DIAGNÓSTICO POR EMOCIÓN ==========")

for indice, emocion in enumerate(classes):

    mascara = (y_true == indice)
    total_reales = int(np.sum(mascara))

    aciertos_keras = int(
        np.sum(keras_pred[mascara] == indice)
    )

    aciertos_tflite = int(
        np.sum(tflite_preds[mascara] == indice)
    )

    recall_keras = (
        aciertos_keras / total_reales * 100
        if total_reales > 0 else 0
    )

    recall_tflite = (
        aciertos_tflite / total_reales * 100
        if total_reales > 0 else 0
    )

    print(f"\nEmoción: {emocion}")
    print(f"Imágenes reales: {total_reales}")
    print(
        f"Recall Keras: {recall_keras:.2f}% "
        f"({aciertos_keras}/{total_reales})"
    )
    print(
        f"Recall TFLite: {recall_tflite:.2f}% "
        f"({aciertos_tflite}/{total_reales})"
    )

print("\nDiagnóstico terminado.")
print("No se entrenó ni sobrescribió ningún modelo.")
