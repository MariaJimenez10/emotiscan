import os
import cv2
import numpy as np
from collections import Counter

# ==========================================================
# TFLITE
# ==========================================================

try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ Usando tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("✅ Usando TensorFlow Lite")


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELO = os.path.join(
    BASE_DIR,
    "modelo_resnet50_emociones.tflite"
)

DATASET_TEST = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

# IMPORTANTE:
# Este debe ser EXACTAMENTE el orden usado durante el entrenamiento.
EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Tristeza",
    "Neutral"
]

IMG_SIZE = 224


# ==========================================================
# CARGAR MODELO
# ==========================================================

print("\n" + "=" * 60)
print("🔄 CARGANDO MODELO")
print("=" * 60)

interpreter = Interpreter(model_path=MODELO)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Modelo cargado")
print("📥 Entrada:", input_details[0]["shape"])
print("📥 Tipo:", input_details[0]["dtype"])
print("📤 Salida:", output_details[0]["shape"])
print("📤 Tipo:", output_details[0]["dtype"])


# ==========================================================
# PREPROCESAMIENTO
# ==========================================================

def preparar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        return None

    # OpenCV BGR -> RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Redimensionar
    imagen = cv2.resize(
        imagen,
        (IMG_SIZE, IMG_SIZE)
    )

    # Convertir a float32
    imagen = imagen.astype(np.float32)

    # ======================================================
    # PREPROCESAMIENTO RESNET50
    # ======================================================

    # RGB -> BGR
    imagen = imagen[:, :, ::-1]

    # Media de ResNet50 / ImageNet
    imagen[:, :, 0] -= 103.939
    imagen[:, :, 1] -= 116.779
    imagen[:, :, 2] -= 123.680

    # Agregar dimensión batch
    imagen = np.expand_dims(
        imagen,
        axis=0
    )

    return imagen.astype(np.float32)


# ==========================================================
# PREDICCIÓN
# ==========================================================

def predecir(ruta):

    imagen = preparar_imagen(ruta)

    if imagen is None:
        return None

    interpreter.set_tensor(
        input_details[0]["index"],
        imagen
    )

    interpreter.invoke()

    salida = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    # Si por alguna razón la salida no suma aproximadamente 1,
    # aplicamos softmax.
    if not np.isclose(
        np.sum(salida),
        1.0,
        atol=0.05
    ):
        exp = np.exp(
            salida - np.max(salida)
        )
        salida = exp / np.sum(exp)

    indice = int(
        np.argmax(salida)
    )

    return indice, salida


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

matriz = np.zeros(
    (len(EMOCIONES), len(EMOCIONES)),
    dtype=int
)


# ==========================================================
# CONTADORES
# ==========================================================

total = 0
correctas = 0

resultados_por_clase = {
    emocion: {
        "total": 0,
        "correctas": 0
    }
    for emocion in EMOCIONES
}


# ==========================================================
# PROBAR DATASET
# ==========================================================

print("\n" + "=" * 60)
print("🧪 PROBANDO DATASET DE TEST")
print("=" * 60)

print("📂", DATASET_TEST)

if not os.path.exists(DATASET_TEST):

    print("\n❌ ERROR:")
    print("No existe la carpeta:")
    print(DATASET_TEST)
    print("\nVerifica que tengas:")
    print("dataset/test/Enojo")
    print("dataset/test/Felicidad")
    print("dataset/test/Tristeza")
    print("dataset/test/Neutral")

    exit()


for indice_real, emocion_real in enumerate(EMOCIONES):

    carpeta = os.path.join(
        DATASET_TEST,
        emocion_real
    )

    if not os.path.exists(carpeta):

        print(
            f"\n⚠️ No existe la carpeta: {carpeta}"
        )

        continue

    archivos = [
        archivo
        for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]

    print(
        f"\n📁 {emocion_real}: {len(archivos)} imágenes"
    )

    for numero, archivo in enumerate(archivos):

        ruta = os.path.join(
            carpeta,
            archivo
        )

        resultado = predecir(ruta)

        if resultado is None:
            continue

        indice_predicho, probabilidades = resultado

        total += 1

        resultados_por_clase[
            emocion_real
        ]["total"] += 1

        matriz[
            indice_real,
            indice_predicho
        ] += 1

        if indice_predicho == indice_real:

            correctas += 1

            resultados_por_clase[
                emocion_real
            ]["correctas"] += 1

        # Mostrar progreso cada 100 imágenes
        if (numero + 1) % 100 == 0:

            print(
                f"   procesadas: {numero + 1}/{len(archivos)}"
            )


# ==========================================================
# RESULTADOS
# ==========================================================

print("\n\n" + "=" * 60)
print("📊 RESULTADOS DEL MODELO")
print("=" * 60)

if total == 0:

    print("❌ No se procesaron imágenes.")
    exit()


accuracy = (
    correctas / total
) * 100


print(
    f"\n🎯 Accuracy general: {accuracy:.2f}%"
)

print(
    f"✅ Correctas: {correctas}/{total}"
)


# ==========================================================
# RESULTADOS POR EMOCIÓN
# ==========================================================

print("\n" + "=" * 60)
print("📌 RESULTADOS POR EMOCIÓN")
print("=" * 60)

for emocion in EMOCIONES:

    datos = resultados_por_clase[
        emocion
    ]

    total_clase = datos["total"]
    correctas_clase = datos["correctas"]

    if total_clase > 0:

        porcentaje = (
            correctas_clase /
            total_clase
        ) * 100

    else:

        porcentaje = 0

    print(
        f"\n{emocion}:"
    )

    print(
        f"   Total: {total_clase}"
    )

    print(
        f"   Correctas: {correctas_clase}"
    )

    print(
        f"   Accuracy: {porcentaje:.2f}%"
    )


# ==========================================================
# MATRIZ DE CONFUSIÓN
# ==========================================================

print("\n" + "=" * 60)
print("📊 MATRIZ DE CONFUSIÓN")
print("=" * 60)

print(
    "\nFilas = emoción REAL"
)

print(
    "Columnas = emoción PREDICHA\n"
)

print(
    "                  " +
    "  ".join(
        f"{e:>10}"
        for e in EMOCIONES
    )
)

for i, emocion in enumerate(EMOCIONES):

    valores = matriz[i]

    print(
        f"{emocion:>15}  " +
        "  ".join(
            f"{valor:>10}"
            for valor in valores
        )
    )


# ==========================================================
# PORCENTAJES DE PREDICCIÓN
# ==========================================================

print("\n" + "=" * 60)
print("📈 DISTRIBUCIÓN DE PREDICCIONES")
print("=" * 60)

totales_predicciones = np.sum(
    matriz,
    axis=0
)

for i, emocion in enumerate(EMOCIONES):

    cantidad = totales_predicciones[i]

    porcentaje = (
        cantidad / total
    ) * 100

    print(
        f"{emocion:10}: "
        f"{cantidad:5} "
        f"({porcentaje:6.2f}%)"
    )


# ==========================================================
# CONCLUSIÓN AUTOMÁTICA
# ==========================================================

print("\n" + "=" * 60)
print("🔎 CONCLUSIÓN")
print("=" * 60)

for emocion in EMOCIONES:

    datos = resultados_por_clase[
        emocion
    ]

    if datos["total"] > 0:

        porcentaje = (
            datos["correctas"] /
            datos["total"]
        ) * 100

        if porcentaje < 30:

            print(
                f"❌ {emocion}: "
                f"muy bajo ({porcentaje:.2f}%)"
            )

        elif porcentaje < 60:

            print(
                f"⚠️ {emocion}: "
                f"bajo/regular ({porcentaje:.2f}%)"
            )

        else:

            print(
                f"✅ {emocion}: "
                f"aceptable ({porcentaje:.2f}%)"
            )

print("\n" + "=" * 60)
print("🏁 PRUEBA TERMINADA")
print("=" * 60)