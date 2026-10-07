import os
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

MODELO = "modelo_resnet50_emociones.tflite"
DATASET_TEST = "dataset/test"

# ==========================================================
# ORDEN DE CLASES
# ==========================================================
#
# Este debe coincidir con el orden usado por el entrenamiento.
#
# Si el modelo fue entrenado leyendo las carpetas de forma
# automática, normalmente Keras utiliza orden alfabético:
#
# 0 = Enojo
# 1 = Felicidad
# 2 = Neutral
# 3 = Tristeza
#
# ==========================================================

CLASES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

IMG_SIZE = 224


# ==========================================================
# CARGAR TFLITE
# ==========================================================

try:

    from tflite_runtime.interpreter import Interpreter

    print("✅ Usando tflite_runtime")

except ImportError:

    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

    print("✅ Usando TensorFlow Lite")


# ==========================================================
# CARGAR MODELO
# ==========================================================

print("\n" + "=" * 60)
print("📦 CARGANDO MODELO")
print("=" * 60)

if not os.path.exists(MODELO):

    raise FileNotFoundError(
        f"No se encontró el modelo:\n{MODELO}"
    )


interpreter = Interpreter(
    model_path=MODELO
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]


print("✅ Modelo cargado correctamente")


print("\n📐 Entrada del modelo:")
print(
    "Shape:",
    input_details[0]["shape"]
)
print(
    "Tipo:",
    input_details[0]["dtype"]
)


print("\n📐 Salida del modelo:")
print(
    "Shape:",
    output_details[0]["shape"]
)
print(
    "Tipo:",
    output_details[0]["dtype"]
)


# ==========================================================
# PREPROCESAMIENTO RESNET50
# ==========================================================

def preparar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:

        return None

    # ------------------------------------------------------
    # OpenCV BGR -> RGB
    # ------------------------------------------------------

    imagen = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------------
    # Redimensionar
    # ------------------------------------------------------

    imagen = cv2.resize(
        imagen,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------
    # Float32
    # ------------------------------------------------------

    imagen = imagen.astype(
        np.float32
    )

    # ------------------------------------------------------
    # PREPROCESAMIENTO CLÁSICO DE RESNET50
    #
    # Equivalente a:
    #
    # tensorflow.keras.applications.resnet50
    # .preprocess_input()
    #
    # RGB -> BGR
    # resta de medias:
    # 103.939
    # 116.779
    # 123.680
    # ------------------------------------------------------

    imagen = imagen[:, :, ::-1]

    imagen[:, :, 0] -= 103.939
    imagen[:, :, 1] -= 116.779
    imagen[:, :, 2] -= 123.680

    # ------------------------------------------------------
    # Agregar dimensión batch
    # ------------------------------------------------------

    imagen = np.expand_dims(
        imagen,
        axis=0
    )

    return imagen


# ==========================================================
# SOFTMAX
# ==========================================================

def softmax(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(imagen):

    # ------------------------------------------------------
    # Verificar tipo de entrada
    # ------------------------------------------------------

    input_dtype = input_details[0]["dtype"]

    if imagen.dtype != input_dtype:

        imagen = imagen.astype(
            input_dtype
        )

    # ------------------------------------------------------
    # Enviar imagen al modelo
    # ------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        imagen
    )

    # ------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------

    interpreter.invoke()

    # ------------------------------------------------------
    # Obtener salida
    # ------------------------------------------------------

    salida = interpreter.get_tensor(
        output_index
    )[0]

    salida = np.asarray(
        salida,
        dtype=np.float32
    )

    # ------------------------------------------------------
    # Normalizar salida
    # ------------------------------------------------------

    suma = float(
        np.sum(salida)
    )

    # Si la salida no parece una distribución
    # de probabilidades, aplicar Softmax.

    if (
        np.any(salida < 0)
        or abs(suma - 1.0) > 0.05
    ):

        salida = softmax(
            salida
        )

    elif suma > 0:

        salida = (
            salida / suma
        )

    # ------------------------------------------------------
    # Predicción
    # ------------------------------------------------------

    prediccion = int(
        np.argmax(salida)
    )

    confianza = float(
        salida[prediccion]
    )

    return (
        prediccion,
        confianza,
        salida
    )


# ==========================================================
# BUSCAR IMÁGENES
# ==========================================================

imagenes = []


print("\n" + "=" * 60)
print("📁 BUSCANDO IMÁGENES DE PRUEBA")
print("=" * 60)


for indice, clase in enumerate(CLASES):

    carpeta = os.path.join(
        DATASET_TEST,
        clase
    )

    if not os.path.exists(carpeta):

        print(
            f"⚠️ No existe la carpeta: {carpeta}"
        )

        continue

    archivos = os.listdir(
        carpeta
    )

    cantidad = 0

    for archivo in archivos:

        ruta = os.path.join(
            carpeta,
            archivo
        )

        if archivo.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        ):

            imagenes.append(
                (
                    ruta,
                    indice
                )
            )

            cantidad += 1

    print(
        f"   {indice}: {clase} → {cantidad} imágenes"
    )


total = len(
    imagenes
)


print(
    f"\n🖼️ Total de imágenes: {total}"
)


# ==========================================================
# EVALUACIÓN
# ==========================================================

y_real = []
y_pred = []

errores = 0


print("\n" + "=" * 60)
print("🚀 INICIANDO EVALUACIÓN")
print("=" * 60)


for contador, (
    ruta,
    etiqueta_real
) in enumerate(
    imagenes,
    start=1
):

    imagen = preparar_imagen(
        ruta
    )

    if imagen is None:

        errores += 1

        print(
            f"⚠️ Imagen inválida: {ruta}"
        )

        continue

    try:

        prediccion, confianza, salida = predecir(
            imagen
        )

        y_real.append(
            etiqueta_real
        )

        y_pred.append(
            prediccion
        )

    except Exception as e:

        errores += 1

        print(
            f"❌ Error procesando {ruta}: {e}"
        )

    # ------------------------------------------------------
    # Mostrar progreso
    # ------------------------------------------------------

    if (
        contador % 100 == 0
        or contador == total
    ):

        porcentaje = (
            contador / total
        ) * 100

        print(
            f"Procesando: "
            f"{contador}/{total} "
            f"({porcentaje:.1f}%)"
        )


# ==========================================================
# VERIFICAR RESULTADOS
# ==========================================================

if len(y_real) == 0:

    raise RuntimeError(
        "No se pudo evaluar ninguna imagen."
    )


# ==========================================================
# MÉTRICAS GENERALES
# ==========================================================

print("\n")
print("=" * 60)
print("📊 MÉTRICAS DEL MODELO")
print("=" * 60)


accuracy = accuracy_score(
    y_real,
    y_pred
)


precision = precision_score(
    y_real,
    y_pred,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_real,
    y_pred,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_real,
    y_pred,
    average="weighted",
    zero_division=0
)


print(
    f"\n🎯 Accuracy : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"🎯 Precision: "
    f"{precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"🎯 Recall   : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"🎯 F1-Score : "
    f"{f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# ==========================================================
# REPORTE POR EMOCIÓN
# ==========================================================

print("\n")
print("=" * 60)
print("📋 REPORTE POR EMOCIÓN")
print("=" * 60)


print(
    classification_report(
        y_real,
        y_pred,
        labels=list(
            range(len(CLASES))
        ),
        target_names=CLASES,
        zero_division=0
    )
)


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

print("=" * 60)
print("🔢 MATRIZ DE CONFUSIÓN")
print("=" * 60)


matriz = confusion_matrix(
    y_real,
    y_pred,
    labels=list(
        range(len(CLASES))
    )
)


print(
    "\n                  "
    + "  ".join(
        f"{c[:10]:>10}"
        for c in CLASES
    )
)


for i, fila in enumerate(
    matriz
):

    print(
        f"{CLASES[i]:>12}      "
        + "  ".join(
            f"{valor:10d}"
            for valor in fila
        )
    )


# ==========================================================
# PREDICCIONES POR CLASE
# ==========================================================

print("\n")
print("=" * 60)
print("🎭 PREDICCIONES REALIZADAS")
print("=" * 60)


for indice, clase in enumerate(
    CLASES
):

    cantidad_real = sum(
        1
        for valor in y_real
        if valor == indice
    )

    cantidad_predicha = sum(
        1
        for valor in y_pred
        if valor == indice
    )

    print(
        f"{clase:>12}: "
        f"real={cantidad_real:<5} "
        f"predicha={cantidad_predicha:<5}"
    )


# ==========================================================
# RESUMEN FINAL
# ==========================================================

print("\n")
print("=" * 60)
print("✅ EVALUACIÓN TERMINADA")
print("=" * 60)


print(
    f"Imágenes evaluadas: "
    f"{len(y_real)}"
)

print(
    f"Errores: "
    f"{errores}"
)


print("\nRESULTADOS FINALES:")


print(
    f"Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score : "
    f"{f1 * 100:.2f}%"
)


print("\n🏁 Listo.")