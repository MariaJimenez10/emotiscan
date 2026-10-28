
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
TEST_DIR = Path("dataset_limpio/test")
CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]

if not MODELO.exists():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

for clase in CLASES:
    if not (TEST_DIR / clase).is_dir():
        raise FileNotFoundError(f"Falta la carpeta: {TEST_DIR / clase}")

modelo = tf.keras.models.load_model(MODELO)

generador = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

datos = generador.flow_from_directory(
    str(TEST_DIR),
    classes=CLASES,
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=False
)

probabilidades = modelo.predict(datos, verbose=1)
predicciones = np.argmax(probabilidades, axis=1)
reales = datos.classes

matriz = confusion_matrix(
    reales, predicciones, labels=list(range(len(CLASES)))
)

print("\nOrden de clases:", CLASES)
print("\nMatriz de confusión:")
print(matriz)

print("\nReporte completo:")
print(classification_report(
    reales,
    predicciones,
    labels=list(range(len(CLASES))),
    target_names=CLASES,
    digits=4,
    zero_division=0
))

total_enojo = int(np.sum(reales == 0))
aciertos_enojo = int(np.sum((reales == 0) & (predicciones == 0)))
recall = aciertos_enojo / total_enojo if total_enojo else 0

print(f"\nTotal real de Enojo: {total_enojo}")
print(f"Clasificadas correctamente como Enojo: {aciertos_enojo}")
print(f"Sensibilidad de Enojo: {recall * 100:.2f}%")
print(
    "Predicciones Enojo en todo test:",
    int(np.sum(predicciones == 0))
)
