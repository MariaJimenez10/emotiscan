
import os
import json
import logging

import cv2
import numpy as np

# ==========================================================
# CONFIGURACIÃ“N
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELO = os.path.join(
    BASE_DIR, "modelo", "modelo_resnet50_emociones.tflite"
)

DATASET_TEST = os.path.join(
    BASE_DIR, "dataset", "test"
)

CLASES_PATH = os.path.join(
    BASE_DIR, "modelo", "clases.json"
)

CARPETA_REVISION = os.path.join(
    BASE_DIR, "revision_errores"
)

IMG_SIZE = 224
MAX_MUESTRAS_POR_CONFUSION = 8
COLUMNAS_LAMINA = 4

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp")

logging.disable(logging.CRITICAL)


# ==========================================================
# VERIFICAR ARCHIVOS Y CLASES
# ==========================================================

print("\n" + "=" * 65)
print("EMOTISCAN - EVALUACIÃ“N Y REVISIÃ“N DE ERRORES")
print("=" * 65)

if not os.path.isfile(MODELO):
    raise FileNotFoundError(
        f"No se encontrÃ³ el modelo:\n{MODELO}"
    )

if not os.path.isdir(DATASET_TEST):
    raise FileNotFoundError(
        f"No se encontrÃ³ el dataset:\n{DATASET_TEST}"
    )

if os.path.isfile(CLASES_PATH):
    with open(CLASES_PATH, "r", encoding="utf-8-sig") as archivo:
        clases_guardadas = json.load(archivo)

    if clases_guardadas != EMOCIONES:
        raise ValueError(
            "El orden de clases no coincide.\n"
            f"Script: {EMOCIONES}\n"
            f"clases.json: {clases_guardadas}"
        )

print(f"\nModelo: {MODELO}")
print(f"Dataset: {DATASET_TEST}")
print(f"Clases: {EMOCIONES}")


# ==========================================================
# CARGAR TENSORFLOW LITE
# ==========================================================

try:
    from tflite_runtime.interpreter import Interpreter
    print("\nUsando tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("\nUsando TensorFlow Lite")

interpreter = Interpreter(model_path=MODELO)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

entrada = input_details[0]
salida = output_details[0]

print("Entrada:", entrada["shape"])
print("Tipo de entrada:", entrada["dtype"])
print("Salida:", salida["shape"])
print("Tipo de salida:", salida["dtype"])

if tuple(entrada["shape"]) != (1, IMG_SIZE, IMG_SIZE, 3):
    raise ValueError(
        f"Entrada inesperada: {entrada['shape']}"
    )

if int(np.prod(salida["shape"])) != len(EMOCIONES):
    raise ValueError(
        f"Salida inesperada: {salida['shape']}"
    )


# ==========================================================
# PREPROCESAMIENTO RESNET50
# ==========================================================

def preparar_imagen(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError(f"No se pudo leer: {ruta}")

    # BGR -> RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    imagen = cv2.resize(
    imagen,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_AREA
)

    imagen = imagen.astype(np.float32)

    # Preprocesamiento ResNet50 en modo Caffe:
    # RGB -> BGR y resta de medias de ImageNet.
    imagen = imagen[..., ::-1].copy()
    imagen[..., 0] -= 103.939
    imagen[..., 1] -= 116.779
    imagen[..., 2] -= 123.680

    imagen = np.expand_dims(imagen, axis=0)

    dtype = entrada["dtype"]

    if dtype == np.float32:
        return imagen.astype(np.float32)

    if dtype == np.float16:
        return imagen.astype(np.float16)

    if dtype in (np.int8, np.uint8):
        escala, cero = entrada["quantization"]

        if escala <= 0:
            raise ValueError(
                "La entrada cuantizada no tiene escala vÃ¡lida."
            )

        cuantizada = np.round(imagen / escala + cero)
        limites = np.iinfo(dtype)

        cuantizada = np.clip(
            cuantizada,
            limites.min,
            limites.max
        )

        return cuantizada.astype(dtype)

    raise TypeError(f"Tipo de entrada no soportado: {dtype}")


# ==========================================================
# PREDICCIÃ“N
# ==========================================================

def predecir(ruta):
    imagen = preparar_imagen(ruta)

    interpreter.set_tensor(entrada["index"], imagen)
    interpreter.invoke()

    valores = interpreter.get_tensor(
        salida["index"]
    )[0].astype(np.float32).reshape(-1)

    if salida["dtype"] in (np.int8, np.uint8):
        escala, cero = salida["quantization"]

        if escala <= 0:
            raise ValueError("Escala de salida no vÃ¡lida.")

        valores = (valores - cero) * escala

    if not np.all(np.isfinite(valores)):
        raise ValueError("El modelo produjo valores no vÃ¡lidos.")

    suma = float(np.sum(valores))

    # Si son probabilidades, conservarlas.
    # Si son logits, aplicar softmax.
    if (
        np.all(valores >= 0)
        and np.all(valores <= 1)
        and np.isclose(suma, 1.0, atol=0.05)
    ):
        probabilidades = valores / suma
    else:
        valores = valores - np.max(valores)
        exp_valores = np.exp(valores)
        probabilidades = exp_valores / np.sum(exp_valores)

    indice = int(np.argmax(probabilidades))

    return indice, probabilidades


# ==========================================================
# EVALUAR EL DATASET Y GUARDAR ERRORES
# ==========================================================

n = len(EMOCIONES)

matriz = np.zeros((n, n), dtype=np.int64)
total = 0
errores_lectura = 0

# Cada clave contiene: (clase real, clase predicha).
# Solo guardamos hasta N ejemplos por combinaciÃ³n.
muestras_errores = {}

print("\n" + "=" * 65)
print("PROBANDO IMÃGENES")
print("=" * 65)

for indice_real, emocion_real in enumerate(EMOCIONES):
    carpeta = os.path.join(DATASET_TEST, emocion_real)

    if not os.path.isdir(carpeta):
        raise FileNotFoundError(
            f"Falta la carpeta de clase: {carpeta}"
        )

    archivos = sorted(
        nombre for nombre in os.listdir(carpeta)
        if nombre.lower().endswith(EXTENSIONES)
        and os.path.isfile(os.path.join(carpeta, nombre))
    )

    print(f"\n{emocion_real}: {len(archivos)} imÃ¡genes")

    for numero, nombre in enumerate(archivos, start=1):
        ruta = os.path.join(carpeta, nombre)

        try:
            indice_predicho, probabilidades = predecir(ruta)
        except (ValueError, TypeError, cv2.error) as error:
            errores_lectura += 1
            print(f"Error con {nombre}: {error}")
            continue

        matriz[indice_real, indice_predicho] += 1
        total += 1

        if indice_real != indice_predicho:
            clave = (indice_real, indice_predicho)

            lista = muestras_errores.setdefault(clave, [])

            if len(lista) < MAX_MUESTRAS_POR_CONFUSION:
                lista.append({
                    "ruta": ruta,
                    "confianza": float(
                        probabilidades[indice_predicho]
                    )
                })

        if numero % 100 == 0 or numero == len(archivos):
            print(f"  Procesadas: {numero}/{len(archivos)}")


# ==========================================================
# RESULTADOS GENERALES
# ==========================================================

print("\n" + "=" * 65)
print("RESULTADOS GENERALES")
print("=" * 65)

if total == 0:
    raise RuntimeError("No se pudo evaluar ninguna imagen.")

correctas = int(np.trace(matriz))
accuracy = 100.0 * correctas / total

print(f"ImÃ¡genes evaluadas: {total}")
print(f"Predicciones correctas: {correctas}")
print(f"Predicciones incorrectas: {total - correctas}")
print(f"Accuracy general: {accuracy:.2f}%")
print(f"Errores de lectura: {errores_lectura}")


# ==========================================================
# RESULTADOS POR EMOCIÃ“N
# ==========================================================

print("\n" + "=" * 65)
print("RESULTADOS POR EMOCIÃ“N")
print("=" * 65)

for i, emocion in enumerate(EMOCIONES):
    total_real = int(np.sum(matriz[i]))
    aciertos = int(matriz[i, i])

    recall = (
        100.0 * aciertos / total_real
        if total_real else 0.0
    )

    print(
        f"{emocion:12} | Total: {total_real:5d} | "
        f"Correctas: {aciertos:5d} | Recall: {recall:6.2f}%"
    )


# ==========================================================
# MATRIZ DE CONFUSIÃ“N
# Filas = emociÃ³n real; columnas = predicciÃ³n
# ==========================================================

print("\n" + "=" * 65)
print("MATRIZ DE CONFUSIÃ“N")
print("=" * 65)

ancho = 12

print(
    f"{'REAL / PRED.':>{ancho}}"
    + "".join(f"{e:>{ancho}}" for e in EMOCIONES)
)

for i, emocion in enumerate(EMOCIONES):
    print(
        f"{emocion:>{ancho}}"
        + "".join(
            f"{int(matriz[i, j]):>{ancho}d}"
            for j in range(n)
        )
    )


# ==========================================================
# DISTRIBUCIÃ“N DE PREDICCIONES
# ==========================================================

print("\n" + "=" * 65)
print("DISTRIBUCIÃ“N DE PREDICCIONES")
print("=" * 65)

totales_predichos = np.sum(matriz, axis=0)

for i, emocion in enumerate(EMOCIONES):
    cantidad = int(totales_predichos[i])
    porcentaje = 100.0 * cantidad / total

    print(
        f"{emocion:12}: {cantidad:5d} ({porcentaje:6.2f}%)"
    )


# ==========================================================
# CREAR LÃMINA DE ERRORES
# ==========================================================

print("\n" + "=" * 65)
print("GENERANDO LÃMINA DE ERRORES")
print("=" * 65)

os.makedirs(CARPETA_REVISION, exist_ok=True)

ANCHO_CELDA = 240
ALTO_CELDA = 190
MARGEN = 8

celdas = []

# Primero muestra las confusiones mÃ¡s numerosas.
combinaciones = sorted(
    muestras_errores.items(),
    key=lambda elemento: matriz[elemento[0][0], elemento[0][1]],
    reverse=True
)

for (indice_real, indice_predicho), elementos in combinaciones:
    real = EMOCIONES[indice_real]
    predicha = EMOCIONES[indice_predicho]
    cantidad_total = int(matriz[indice_real, indice_predicho])

    print(
        f"{real} -> {predicha}: {cantidad_total} errores; "
        f"muestra guardada: {len(elementos)}"
    )

    for elemento in elementos:
        ruta = elemento["ruta"]
        confianza = elemento["confianza"]

        imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)

        if imagen is None:
            continue

        alto, ancho = imagen.shape[:2]

        escala = min(
            (ANCHO_CELDA - 2 * MARGEN) / ancho,
            (ALTO_CELDA - 62) / alto
        )

        nuevo_ancho = max(1, int(ancho * escala))
        nuevo_alto = max(1, int(alto * escala))

        miniatura = cv2.resize(
            imagen,
            (nuevo_ancho, nuevo_alto),
            interpolation=cv2.INTER_AREA
        )

        celda = np.full(
            (ALTO_CELDA, ANCHO_CELDA, 3),
            255,
            dtype=np.uint8
        )

        x = (ANCHO_CELDA - nuevo_ancho) // 2
        y = 42 + max(
            0,
            (ALTO_CELDA - 62 - nuevo_alto) // 2
        )

        celda[y:y + nuevo_alto, x:x + nuevo_ancho] = miniatura

        nombre = os.path.basename(ruta)
        if len(nombre) > 29:
            nombre = nombre[:26] + "..."

        cv2.putText(
            celda, f"Real: {real}", (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (20, 20, 20), 1, cv2.LINE_AA
        )

        cv2.putText(
            celda, f"Predicha: {predicha}", (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (20, 20, 20), 1, cv2.LINE_AA
        )

        cv2.putText(
            celda, nombre, (6, ALTO_CELDA - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (20, 20, 20), 1, cv2.LINE_AA
        )

        cv2.putText(
            celda, f"Confianza: {confianza * 100:.1f}%",
            (6, ALTO_CELDA - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (50, 50, 50), 1, cv2.LINE_AA
        )

        celdas.append(celda)

if celdas:
    columnas = COLUMNAS_LAMINA
    filas = (len(celdas) + columnas - 1) // columnas

    lamina = np.full(
        (filas * ALTO_CELDA, columnas * ANCHO_CELDA, 3),
        245,
        dtype=np.uint8
    )

    for i, celda in enumerate(celdas):
        fila = i // columnas
        columna = i % columnas

        y0 = fila * ALTO_CELDA
        x0 = columna * ANCHO_CELDA

        lamina[
            y0:y0 + ALTO_CELDA,
            x0:x0 + ANCHO_CELDA
        ] = celda

    ruta_lamina = os.path.join(
        CARPETA_REVISION,
        "muestra_errores.jpg"
    )

    if cv2.imwrite(ruta_lamina, lamina):
        print(f"\nLÃ¡mina creada correctamente:\n{ruta_lamina}")
        print(f"ImÃ¡genes incluidas: {len(celdas)}")
    else:
        print("No se pudo guardar la lÃ¡mina.")
else:
    print("No se encontraron imÃ¡genes para la lÃ¡mina.")


# ==========================================================
# FINAL
# ==========================================================

print("\n" + "=" * 65)
print("EVALUACIÃ“N TERMINADA")
print("=" * 65)
print("No se entrenÃ³ ni se modificÃ³ el modelo.")
print("No se modificaron las imÃ¡genes originales del dataset.")

