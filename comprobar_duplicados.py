
import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import predict


# ==========================================================
# CONFIGURACION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "test")

KERAS_PATH = os.path.join(
    BASE_DIR, "modelo", "resnet50_emociones_mejorado.keras"
)

EMOCIONES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]


# ==========================================================
# CARGAR LOS DOS MODELOS
# ==========================================================

print("=" * 65)
print("COMPARACION CON IMAGENES REALES: KERAS VS TFLITE")
print("=" * 65)

if not os.path.isfile(KERAS_PATH):
    raise FileNotFoundError(f"No existe el modelo Keras: {KERAS_PATH}")

print("\nCargando modelo Keras...")
modelo_keras = tf.keras.models.load_model(KERAS_PATH, compile=False)

print("Cargando modelo TFLite de EmotiScan...")
predict.cargar_modelo()

if predict.interpreter is None:
    raise RuntimeError("No se pudo cargar el modelo TFLite.")

if list(predict.EMOCIONES) != EMOCIONES:
    raise ValueError("El orden de las clases no coincide con predict.py.")

detalles_entrada = predict.input_details[0]
detalles_salida = predict.output_details[0]

print("Keras:", KERAS_PATH)
print("TFLite:", predict.MODEL_PATH)
print("Clases:", EMOCIONES)


# ==========================================================
# EVALUAR LAS MISMAS IMAGENES EN AMBOS MODELOS
# ==========================================================

y_real = []
pred_keras = []
pred_tflite = []

diferencias = []
errores = 0

for indice, emocion in enumerate(EMOCIONES):
    carpeta = os.path.join(DATASET_PATH, emocion)

    if not os.path.isdir(carpeta):
        print(f"ADVERTENCIA: falta la carpeta {carpeta}")
        continue

    archivos = sorted([
        nombre for nombre in os.listdir(carpeta)
        if nombre.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    print(f"\nEvaluando {emocion}: {len(archivos)} imágenes")

    for nombre in archivos:
        ruta = os.path.join(carpeta, nombre)
        imagen = cv2.imread(ruta)

        if imagen is None:
            errores += 1
            print("No se pudo leer:", ruta)
            continue

        try:
            # Mismo preprocesamiento que usa la aplicación.
            entrada = predict.preparar_rostro(imagen)
            entrada = np.asarray(entrada, dtype=np.float32)

            if entrada.shape != tuple(detalles_entrada["shape"]):
                raise ValueError(
                    f"Forma de entrada inesperada: {entrada.shape}; "
                    f"se esperaba {detalles_entrada['shape']}"
                )

            # Predicción Keras.
            salida_keras = modelo_keras(
                entrada, training=False
            ).numpy().reshape(-1)

            # Predicción TFLite.
            dtype_tflite = detalles_entrada["dtype"]

            if dtype_tflite in (np.int8, np.uint8):
                escala, cero = detalles_entrada["quantization"]

                if escala <= 0:
                    raise ValueError("Escala de cuantización inválida.")

                entrada_tflite = np.clip(
                    np.round(entrada / escala + cero),
                    np.iinfo(dtype_tflite).min,
                    np.iinfo(dtype_tflite).max
                ).astype(dtype_tflite)
            else:
                entrada_tflite = entrada.astype(dtype_tflite)

            predict.interpreter.set_tensor(
                detalles_entrada["index"], entrada_tflite
            )
            predict.interpreter.invoke()

            salida_tflite = predict.interpreter.get_tensor(
                detalles_salida["index"]
            ).astype(np.float64).reshape(-1)

            # Si la salida es cuantizada, convertirla a valores reales.
            if detalles_salida["dtype"] in (np.int8, np.uint8):
                escala_s, cero_s = detalles_salida["quantization"]
                if escala_s <= 0:
                    raise ValueError("Escala de salida inválida.")
                salida_tflite = (salida_tflite - cero_s) * escala_s

            if len(salida_keras) != 4 or len(salida_tflite) != 4:
                raise ValueError("Uno de los modelos no devolvió 4 valores.")

            clase_keras = int(np.argmax(salida_keras))
            clase_tflite = int(np.argmax(salida_tflite))

            y_real.append(indice)
            pred_keras.append(clase_keras)
            pred_tflite.append(clase_tflite)

            diferencias.append(
                float(np.max(np.abs(salida_keras - salida_tflite)))
            )

        except Exception as error:
            errores += 1
            print(f"Error en {ruta}: {error}")


# ==========================================================
# MOSTRAR RESULTADOS
# ==========================================================

if not y_real:
    raise RuntimeError("No se pudo evaluar ninguna imagen.")

acc_keras = accuracy_score(y_real, pred_keras)
acc_tflite = accuracy_score(y_real, pred_tflite)

desacuerdos = sum(
    a != b for a, b in zip(pred_keras, pred_tflite)
)

print("\n" + "=" * 65)
print("RESUMEN DE LA COMPARACION")
print("=" * 65)
print("Imágenes evaluadas:", len(y_real))
print("Errores de lectura/procesamiento:", errores)
print(f"Accuracy Keras:  {acc_keras * 100:.2f}%")
print(f"Accuracy TFLite: {acc_tflite * 100:.2f}%")
print("Predicciones diferentes:", desacuerdos)
print(f"Porcentaje de desacuerdo: {desacuerdos / len(y_real) * 100:.2f}%")
print(
    "Diferencia máxima entre salidas:",
    f"{max(diferencias):.10f}" if diferencias else "No disponible"
)

print("\nREPORTE KERAS")
print(classification_report(
    y_real, pred_keras,
    labels=list(range(4)),
    target_names=EMOCIONES,
    digits=4,
    zero_division=0
))

print("\nREPORTE TFLITE")
print(classification_report(
    y_real, pred_tflite,
    labels=list(range(4)),
    target_names=EMOCIONES,
    digits=4,
    zero_division=0
))

print("\nMATRIZ DE CONFUSION KERAS")
print("Filas = emoción real; columnas = predicción")
print(confusion_matrix(y_real, pred_keras, labels=list(range(4))))

print("\nMATRIZ DE CONFUSION TFLITE")
print("Filas = emoción real; columnas = predicción")
print(confusion_matrix(y_real, pred_tflite, labels=list(range(4))))

print("\nComparación finalizada.")