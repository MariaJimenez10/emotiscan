import os
import shutil
import csv
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


# =========================
# CONFIGURACIÓN
# =========================

MODELO_PATH = "modelo_resnet50_emociones.h5"
CARPETA_ENOJO = "dataset/test/Enojo"
CARPETA_SALIDA = "errores_enojo"

IMG_SIZE = 224

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]


# =========================
# CARGAR MODELO
# =========================

print("Cargando modelo...")

model = load_model(
    MODELO_PATH,
    compile=False
)

print("Modelo cargado correctamente.")


# =========================
# CREAR CARPETAS
# =========================

for emocion in EMOCIONES:
    os.makedirs(
        os.path.join(CARPETA_SALIDA, emocion),
        exist_ok=True
    )


# =========================
# ARCHIVO CSV
# =========================

csv_path = os.path.join(
    CARPETA_SALIDA,
    "reporte_enojo.csv"
)

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "archivo",
    "emocion_real",
    "emocion_predicha",
    "confianza",
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
])


# =========================
# CONTADORES
# =========================

total = 0
correctas = 0
errores = 0


# =========================
# ANALIZAR IMÁGENES
# =========================

archivos = [
    f for f in os.listdir(CARPETA_ENOJO)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print()
print(f"Imágenes encontradas: {len(archivos)}")
print()
print("Analizando...")
print("-" * 70)


for i, archivo in enumerate(archivos, start=1):

    ruta = os.path.join(
        CARPETA_ENOJO,
        archivo
    )

    try:

        # Cargar imagen
        img = load_img(
            ruta,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img_array = img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # IMPORTANTE:
        # mismo preprocesamiento usado durante entrenamiento
        img_array = preprocess_input(img_array)

        # Predicción
        pred = model.predict(
            img_array,
            verbose=0
        )[0]

        indice = np.argmax(pred)

        emocion_predicha = EMOCIONES[indice]

        confianza = float(pred[indice]) * 100

        # Probabilidades
        p_enojo = float(pred[0]) * 100
        p_felicidad = float(pred[1]) * 100
        p_neutral = float(pred[2]) * 100
        p_tristeza = float(pred[3]) * 100

        total += 1

        # =========================
        # CORRECTA
        # =========================

        if emocion_predicha == "Enojo":

            correctas += 1

        # =========================
        # ERROR
        # =========================

        else:

            errores += 1

            carpeta_error = os.path.join(
                CARPETA_SALIDA,
                emocion_predicha
            )

            destino = os.path.join(
                carpeta_error,
                archivo
            )

            shutil.copy2(
                ruta,
                destino
            )

            print(
                f"[ERROR] {archivo} -> "
                f"{emocion_predicha} "
                f"({confianza:.2f}%)"
            )

            writer.writerow([
                archivo,
                "Enojo",
                emocion_predicha,
                f"{confianza:.2f}",
                f"{p_enojo:.2f}",
                f"{p_felicidad:.2f}",
                f"{p_neutral:.2f}",
                f"{p_tristeza:.2f}"
            ])

        # Progreso cada 50 imágenes
        if i % 50 == 0:

            print(
                f"\nProgreso: {i}/{len(archivos)}\n"
            )

    except Exception as e:

        print(
            f"Error procesando {archivo}: {e}"
        )


csv_file.close()


# =========================
# RESULTADOS
# =========================

print()
print("=" * 70)
print("RESULTADO FINAL")
print("=" * 70)

print(f"Total imágenes de Enojo: {total}")
print(f"Correctamente clasificadas: {correctas}")
print(f"Mal clasificadas: {errores}")

if total > 0:

    recall = (correctas / total) * 100

    print(
        f"Recall de Enojo: {recall:.2f}%"
    )

print()
print("Las imágenes incorrectas fueron copiadas a:")

print(
    os.path.abspath(CARPETA_SALIDA)
)

print()
print("Estructura creada:")

print("""
errores_enojo/
│
├── Enojo/
├── Felicidad/
├── Neutral/
├── Tristeza/
│
└── reporte_enojo.csv
""")

print("=" * 70)