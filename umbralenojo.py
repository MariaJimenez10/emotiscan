from pathlib import Path
import csv
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# =====================================================
# CONFIGURACIÓN: SOLO EVALUACIÓN, SIN MODIFICAR MODELOS
# =====================================================

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
CARPETA_TEST = Path("dataset_limpio/test")
SALIDA_CSV = Path("resultados_umbrales_enojo.csv")

CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
INDICE_ENOJO = 0
TAMANO = (224, 224)
BATCH_SIZE = 16

# Umbrales que vamos a probar para la clase Enojo.
UMBRALES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

if not MODELO.exists():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not CARPETA_TEST.exists():
    raise FileNotFoundError(f"No existe el conjunto de prueba: {CARPETA_TEST}")

print("\n======================================")
print(" EMOTISCAN - PRUEBA DE UMBRALES ENOJO")
print("======================================")
print("Modelo:", MODELO.resolve())

modelo = tf.keras.models.load_model(str(MODELO))

generador = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    str(CARPETA_TEST),
    classes=CLASES,
    target_size=TAMANO,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    interpolation="nearest"
)

generador.reset()
probabilidades = modelo.predict(generador, verbose=1)
y_real = generador.classes

if probabilidades.shape[1] != len(CLASES):
    raise ValueError(
        f"El modelo devuelve {probabilidades.shape[1]} clases, "
        f"pero se esperaban {len(CLASES)}."
    )

# Predicción original: clase con mayor probabilidad.
y_original = np.argmax(probabilidades, axis=1)

print("\n--- RESULTADO ORIGINAL ---")
print(classification_report(
    y_real,
    y_original,
    labels=list(range(len(CLASES))),
    target_names=CLASES,
    digits=4,
    zero_division=0
))

resultados = []

for umbral in UMBRALES:
    # Si Enojo supera el umbral, se asigna Enojo.
    # Si no lo supera, se conserva la mejor clase entre las otras tres.
    y_ajustado = y_original.copy()
    no_enojo = probabilidades[:, INDICE_ENOJO] < umbral

    probabilidades_otros = probabilidades[:, 1:]
    y_ajustado[no_enojo] = (
        np.argmax(probabilidades_otros[no_enojo], axis=1) + 1
    )
    y_ajustado[~no_enojo] = INDICE_ENOJO

    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_real,
        y_ajustado,
        labels=list(range(len(CLASES))),
        zero_division=0
    )

    matriz = confusion_matrix(
        y_real,
        y_ajustado,
        labels=list(range(len(CLASES)))
    )

    exactitud = float(np.mean(y_ajustado == y_real))

    fila = {
        "umbral_enojo": umbral,
        "exactitud_general": exactitud,
        "precision_enojo": precision[0],
        "sensibilidad_enojo": recall[0],
        "f1_enojo": f1[0],
        "predicciones_enojo": int(np.sum(y_ajustado == 0)),
        "falsos_positivos_enojo": int(matriz[1:, 0].sum()),
        "precision_felicidad": precision[1],
        "sensibilidad_felicidad": recall[1],
        "f1_felicidad": f1[1],
        "precision_tristeza": precision[2],
        "sensibilidad_tristeza": recall[2],
        "f1_tristeza": f1[2],
        "precision_neutral": precision[3],
        "sensibilidad_neutral": recall[3],
        "f1_neutral": f1[3],
    }

    resultados.append(fila)

print("\n--- COMPARACIÓN DE UMBRALES ---")
print(
    f"{'Umbral':>8} {'Exactitud':>10} {'Prec. Enojo':>12} "
    f"{'Recall Enojo':>13} {'F1 Enojo':>10} {'Pred. Enojo':>12} "
    f"{'Falsas alarmas':>15}"
)

for r in resultados:
    print(
        f"{r['umbral_enojo']:>7.0%} "
        f"{r['exactitud_general']:>9.2%} "
        f"{r['precision_enojo']:>11.2%} "
        f"{r['sensibilidad_enojo']:>12.2%} "
        f"{r['f1_enojo']:>9.2%} "
        f"{r['predicciones_enojo']:>12} "
        f"{r['falsos_positivos_enojo']:>15}"
    )

with SALIDA_CSV.open("w", newline="", encoding="utf-8-sig") as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(resultados[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resultados)

print("\nCSV guardado en:", SALIDA_CSV.resolve())
print("\nNo se modificó ningún modelo ni archivo de EmotiScan.")
print(
    "\nIMPORTANTE: estos umbrales son exploratorios. "
    "El conjunto de prueba no debe utilizarse para elegir "
    "el umbral final de producción."
)