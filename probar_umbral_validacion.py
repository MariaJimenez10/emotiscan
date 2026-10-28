from pathlib import Path
import csv
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

# Configuración idéntica a la separación del entrenamiento
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
TRAIN_DIR = Path("dataset_limpio/train")
SALIDA = Path("resultados_umbrales_validacion.csv")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if not MODELO.is_file():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"No existe la carpeta: {TRAIN_DIR}")

def obtener_imagenes(carpeta):
    rutas = []
    etiquetas = []

    for indice, clase in enumerate(CLASSES):
        subcarpeta = carpeta / clase

        if not subcarpeta.is_dir():
            raise FileNotFoundError(f"Falta la carpeta: {subcarpeta}")

        archivos = sorted(
            archivo
            for archivo in subcarpeta.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONS
        )

        rutas.extend(str(archivo.resolve()) for archivo in archivos)
        etiquetas.extend([indice] * len(archivos))

    return np.array(rutas), np.array(etiquetas, dtype=np.int32)

print("\n=== VALIDACIÓN DE UMBRALES PARA EMOTISCAN ===")
print("Cargando rutas de entrenamiento...")

rutas_train, etiquetas_train = obtener_imagenes(TRAIN_DIR)

# Esta separación reproduce la utilizada en resnet50_limpio.py
_, rutas_val, _, etiquetas_val = train_test_split(
    rutas_train,
    etiquetas_train,
    test_size=0.15,
    random_state=SEED,
    stratify=etiquetas_train
)

print("Imágenes de validación:", len(rutas_val))

# Mismo preprocesamiento y tamaño de imagen que el entrenamiento.
generador_imagenes = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

df_val = pd.DataFrame({
    "filename": rutas_val,
    "class": [CLASSES[int(i)] for i in etiquetas_val]
})

generador = generador_imagenes.flow_from_dataframe(
    dataframe=df_val,
    x_col="filename",
    y_col="class",
    classes=CLASSES,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    interpolation="nearest"
)

print("Cargando modelo experimental...")
modelo = tf.keras.models.load_model(str(MODELO))

generador.reset()
probabilidades = modelo.predict(generador, verbose=1)
y_real = generador.classes

if probabilidades.shape[1] != len(CLASSES):
    raise ValueError("La cantidad de salidas del modelo no coincide con las clases.")

# Verificar el orden de etiquetas
if generador.class_indices != {c: i for i, c in enumerate(CLASSES)}:
    raise ValueError(f"Orden de clases inesperado: {generador.class_indices}")

# Predicción estándar: mayor probabilidad
y_original = np.argmax(probabilidades, axis=1)

p, r, f, _ = precision_recall_fscore_support(
    y_real,
    y_original,
    labels=list(range(len(CLASSES))),
    zero_division=0
)

print("\n=== RESULTADO BASE EN VALIDACIÓN ===")
print(f"Exactitud: {accuracy_score(y_real, y_original):.2%}")
print(f"Precisión Enojo: {p[0]:.2%}")
print(f"Sensibilidad Enojo: {r[0]:.2%}")
print(f"F1 Enojo: {f[0]:.2%}")

umbrales = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
resultados = []

for umbral in umbrales:
    # Regla experimental:
    # si la probabilidad de Enojo alcanza el umbral, se predice Enojo;
    # de lo contrario, se elige entre las otras tres emociones.
    y_pred = np.argmax(probabilidades[:, 1:], axis=1) + 1
    es_enojo = probabilidades[:, 0] >= umbral
    y_pred[es_enojo] = 0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0
    )

    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=list(range(len(CLASSES)))
    )

    resultados.append({
        "umbral": umbral,
        "exactitud_general": accuracy_score(y_real, y_pred),
        "precision_enojo": precision[0],
        "sensibilidad_enojo": recall[0],
        "f1_enojo": f1[0],
        "predicciones_enojo": int(np.sum(y_pred == 0)),
        "falsas_alarmas_enojo": int(matriz[1:, 0].sum()),
        "recall_felicidad": recall[1],
        "recall_tristeza": recall[2],
        "recall_neutral": recall[3]
    })

print("\n=== COMPARACIÓN EN VALIDACIÓN ===")
print(
    f"{'Umbral':>7} {'Exactitud':>10} {'Precisión':>11} "
    f"{'Sensibilidad':>13} {'F1 Enojo':>10} "
    f"{'Falsas alarmas':>15}"
)

for fila in resultados:
    print(
        f"{fila['umbral']:>6.0%} "
        f"{fila['exactitud_general']:>9.2%} "
        f"{fila['precision_enojo']:>10.2%} "
        f"{fila['sensibilidad_enojo']:>12.2%} "
        f"{fila['f1_enojo']:>9.2%} "
        f"{fila['falsas_alarmas_enojo']:>15}"
    )

with SALIDA.open("w", newline="", encoding="utf-8-sig") as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(resultados[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("\nResultados guardados en:", SALIDA.resolve())
print("No se modificaron el modelo ni la aplicación.")
