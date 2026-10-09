
import os
import cv2
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import predict


# ==========================================================
# CONFIGURACION
# ==========================================================

DATASET_PATH = os.path.join("dataset", "test")

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# ==========================================================
# CARGAR MODELO TFLITE DE LA APLICACION
# ==========================================================

print("=" * 60)
print("EVALUACION DEL MODELO TFLITE DE EMOTISCAN AI")
print("=" * 60)

predict.cargar_modelo()

if predict.interpreter is None:
    raise RuntimeError("No se pudo cargar el modelo TFLite.")

print("Modelo cargado desde:", predict.MODEL_PATH)
print("Clases:", predict.EMOCIONES)

if list(predict.EMOCIONES) != EMOCIONES:
    raise ValueError(
        "El orden de las emociones no coincide con predict.py."
    )


# ==========================================================
# PREPARAR IMAGENES Y ETIQUETAS
# ==========================================================

y_real = []
y_predicho = []
errores_lectura = 0

for indice, emocion in enumerate(EMOCIONES):

    carpeta = os.path.join(DATASET_PATH, emocion)

    if not os.path.isdir(carpeta):
        print(f"ADVERTENCIA: no existe la carpeta {carpeta}")
        continue

    archivos = [
        archivo for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    print(f"{emocion}: {len(archivos)} imágenes")

    for archivo in archivos:

        ruta = os.path.join(carpeta, archivo)
        imagen = cv2.imread(ruta)

        if imagen is None:
            errores_lectura += 1
            continue

        try:
            # Usar el mismo preprocesamiento que predict.py
            entrada = predict.preparar_rostro(imagen)

            detalles_entrada = predict.input_details[0]
            dtype = detalles_entrada["dtype"]

            # Adaptar la entrada al tipo esperado por TFLite
            if dtype in (np.int8, np.uint8):
                escala, cero = detalles_entrada["quantization"]

                if escala <= 0:
                    raise ValueError(
                        "Cuantización de entrada inválida."
                    )

                entrada = np.round(
                    entrada / escala + cero
                )

                limites = np.iinfo(dtype)
                entrada = np.clip(
                    entrada,
                    limites.min,
                    limites.max
                ).astype(dtype)

            else:
                entrada = entrada.astype(dtype)

            predict.interpreter.set_tensor(
                detalles_entrada["index"],
                entrada
            )

            predict.interpreter.invoke()

            salida = predict.interpreter.get_tensor(
                predict.output_details[0]["index"]
            )

            salida = np.asarray(
                salida,
                dtype=np.float64
            ).reshape(-1)

            if len(salida) != len(EMOCIONES):
                raise ValueError(
                    f"El modelo devolvió {len(salida)} valores "
                    f"en lugar de {len(EMOCIONES)}."
                )

            # Convertir la salida a probabilidades si es necesario
            if (
                np.all(salida >= 0)
                and np.isclose(np.sum(salida), 1.0, atol=0.02)
            ):
                probabilidades = salida / np.sum(salida)
            else:
                salida = salida - np.max(salida)
                exp_salida = np.exp(salida)
                probabilidades = exp_salida / np.sum(exp_salida)

            prediccion = int(np.argmax(probabilidades))

            y_real.append(indice)
            y_predicho.append(prediccion)

        except Exception as error:
            print(f"Error con {ruta}: {error}")
            errores_lectura += 1


# ==========================================================
# RESULTADOS
# ==========================================================

if not y_real:
    raise RuntimeError(
        "No se evaluaron imágenes. Revisa la ruta dataset/test "
        "y las carpetas de emociones."
    )

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("Imágenes evaluadas:", len(y_real))
print("Imágenes con errores:", errores_lectura)

accuracy = accuracy_score(y_real, y_predicho)

print(f"\nACCURACY: {accuracy * 100:.2f}%")

print("\nREPORTE POR EMOCION")
print(
    classification_report(
        y_real,
        y_predicho,
        labels=list(range(len(EMOCIONES))),
        target_names=EMOCIONES,
        digits=4,
        zero_division=0
    )
)

matriz = confusion_matrix(
    y_real,
    y_predicho,
    labels=list(range(len(EMOCIONES)))
)

print("MATRIZ DE CONFUSION")
print("Filas = emoción real; columnas = emoción predicha")
print("Orden:", EMOCIONES)
print(matriz)

with open(
    "resultado_evaluacion_tflite.txt",
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write("EVALUACION DEL MODELO TFLITE\n")
    archivo.write(f"Modelo: {predict.MODEL_PATH}\n")
    archivo.write(f"Imagenes evaluadas: {len(y_real)}\n")
    archivo.write(f"Errores de lectura/procesamiento: {errores_lectura}\n")
    archivo.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")

    archivo.write("REPORTE POR EMOCION\n")
    archivo.write(
        classification_report(
            y_real,
            y_predicho,
            labels=list(range(len(EMOCIONES))),
            target_names=EMOCIONES,
            digits=4,
            zero_division=0
        )
    )

    archivo.write("\nMATRIZ DE CONFUSION\n")
    archivo.write(
        "Filas = emoción real; columnas = emoción predicha\n"
    )
    archivo.write(f"Orden: {EMOCIONES}\n")
    archivo.write(np.array2string(matriz))

print("\nResultados guardados en resultado_evaluacion_tflite.txt")
print("Evaluación finalizada.")