
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
TEST_DIR = Path("dataset_limpio/test")

CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
TAMANO = (224, 224)
BATCH_SIZE = 16
UMBRAL_ENOJO = 0.30

if not MODELO.is_file():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"No existe el conjunto de prueba: {TEST_DIR}")

generador_imagenes = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

generador = generador_imagenes.flow_from_directory(
    str(TEST_DIR),
    classes=CLASES,
    target_size=TAMANO,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    interpolation="nearest"
)

modelo = tf.keras.models.load_model(str(MODELO))

generador.reset()
probabilidades = modelo.predict(generador, verbose=1)
y_real = generador.classes

if probabilidades.shape[1] != len(CLASES):
    raise ValueError("El modelo no devuelve cuatro clases.")

# ComparaciÃ³n: predicciÃ³n original
y_original = np.argmax(probabilidades, axis=1)

# PredicciÃ³n experimental con umbral de Enojo del 30 %
y_umbral = np.argmax(probabilidades[:, 1:], axis=1) + 1
y_umbral[probabilidades[:, 0] >= UMBRAL_ENOJO] = 0

for nombre, predicciones in [
    ("ORIGINAL", y_original),
    ("UMBRAL ENOJO 30%", y_umbral)
]:
    print("\n" + "=" * 55)
    print(nombre)
    print("=" * 55)

    print(f"Exactitud general: {accuracy_score(y_real, predicciones):.2%}")

    print("\nReporte por emociÃ³n:")
    print(classification_report(
        y_real,
        predicciones,
        labels=list(range(len(CLASES))),
        target_names=CLASES,
        digits=4,
        zero_division=0
    ))

    print("Matriz de confusiÃ³n:")
    print("Orden:", CLASES)
    print(confusion_matrix(
        y_real,
        predicciones,
        labels=list(range(len(CLASES)))
    ))

print("\nEvaluaciÃ³n terminada. No se modificÃ³ ningÃºn modelo ni la aplicaciÃ³n.")

