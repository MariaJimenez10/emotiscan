import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODEL_PATH = "modelo_resnet50_emociones.h5"
TEST_PATH = "dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 16

EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

print("=" * 70)
print("ANÁLISIS DE PREDICCIONES DEL MODELO RESNET50")
print("=" * 70)


# ============================================================
# 1. VERIFICAR MODELO
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"\n❌ No se encontró el modelo:")
    print(MODEL_PATH)
    exit()

print("\n✅ Modelo encontrado:")
print(MODEL_PATH)


# ============================================================
# 2. CARGAR MODELO
# ============================================================

print("\nCargando modelo...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Modelo cargado correctamente")


# ============================================================
# 3. CARGAR TEST
# ============================================================

print("\nPreparando TEST...")

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=EMOCIONES,
    shuffle=False
)

print("\nClases detectadas:")
print(test_generator.class_indices)

print(f"\nImágenes TEST: {test_generator.samples}")


# ============================================================
# 4. REALIZAR PREDICCIONES
# ============================================================

print("\n" + "=" * 70)
print("REALIZANDO PREDICCIONES")
print("=" * 70)

predicciones_prob = model.predict(
    test_generator,
    verbose=1
)

predicciones = np.argmax(predicciones_prob, axis=1)

verdaderas = test_generator.classes


# ============================================================
# 5. ACCURACY
# ============================================================

accuracy = accuracy_score(verdaderas, predicciones)

print("\n" + "=" * 70)
print("RESULTADO GENERAL")
print("=" * 70)

print(f"\nAccuracy TEST: {accuracy * 100:.2f}%")


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        verdaderas,
        predicciones,
        target_names=EMOCIONES,
        digits=4
    )
)


# ============================================================
# 7. MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    verdaderas,
    predicciones
)

print("\n" + "=" * 70)
print("MATRIZ DE CONFUSIÓN")
print("=" * 70)

print("\nFilas = emoción real")
print("Columnas = emoción predicha\n")

print("             ", end="")

for emocion in EMOCIONES:
    print(f"{emocion:>12}", end="")

print()

for i, emocion in enumerate(EMOCIONES):

    print(f"{emocion:<12}", end="")

    for j in range(len(EMOCIONES)):
        print(f"{cm[i][j]:>12}", end="")

    print()


# ============================================================
# 8. ANÁLISIS POR EMOCIÓN
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS POR EMOCIÓN")
print("=" * 70)

for i, emocion in enumerate(EMOCIONES):

    indices = np.where(verdaderas == i)[0]

    total = len(indices)

    aciertos = np.sum(predicciones[indices] == i)

    errores = total - aciertos

    accuracy_emocion = (
        aciertos / total
        if total > 0
        else 0
    )

    confianza_promedio = np.mean(
        np.max(predicciones_prob[indices], axis=1)
    )

    confianza_clase_real = np.mean(
        predicciones_prob[indices, i]
    )

    print(f"\n{emocion}")
    print("-" * 40)

    print(f"Total imágenes:             {total}")
    print(f"Aciertos:                   {aciertos}")
    print(f"Errores:                    {errores}")
    print(f"Recall:                     {accuracy_emocion * 100:.2f}%")
    print(f"Confianza promedio:         {confianza_promedio * 100:.2f}%")
    print(
        f"Confianza en clase real:    "
        f"{confianza_clase_real * 100:.2f}%"
    )

    print("\nPredicciones realizadas:")

    for j, pred_emocion in enumerate(EMOCIONES):

        cantidad = np.sum(
            predicciones[indices] == j
        )

        porcentaje = (
            cantidad / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {pred_emocion:<12}: "
            f"{cantidad:4d} "
            f"({porcentaje:6.2f}%)"
        )


# ============================================================
# 9. ANALIZAR ESPECÍFICAMENTE ENOJO
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS ESPECÍFICO DE ENOJO")
print("=" * 70)

enojo_indices = np.where(
    verdaderas == 0
)[0]

print(f"\nTotal imágenes de ENOJO: {len(enojo_indices)}")

for i, emocion in enumerate(EMOCIONES):

    cantidad = np.sum(
        predicciones[enojo_indices] == i
    )

    porcentaje = (
        cantidad / len(enojo_indices) * 100
    )

    print(
        f"ENOJO → {emocion:<12}: "
        f"{cantidad:4d} "
        f"({porcentaje:6.2f}%)"
    )


# ============================================================
# 10. CREAR REPORTE DETALLADO
# ============================================================

print("\n" + "=" * 70)
print("CREANDO REPORTE DETALLADO")
print("=" * 70)

filenames = test_generator.filenames

datos = []

for i in range(len(verdaderas)):

    fila = {
        "archivo": filenames[i],
        "emocion_real": EMOCIONES[verdaderas[i]],
        "emocion_predicha": EMOCIONES[predicciones[i]],
        "correcta": verdaderas[i] == predicciones[i],
        "confianza_prediccion": float(
            np.max(predicciones_prob[i])
        )
    }

    # Probabilidad de cada emoción
    for j, emocion in enumerate(EMOCIONES):

        fila[
            f"prob_{emocion}"
        ] = float(
            predicciones_prob[i][j]
        )

    datos.append(fila)


df = pd.DataFrame(datos)


# ============================================================
# 11. GUARDAR CSV
# ============================================================

CSV_PATH = "reporte_predicciones_resnet50.csv"

df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ Reporte guardado:")
print(CSV_PATH)


# ============================================================
# 12. TOP 20 ERRORES MÁS SEGUROS
# ============================================================

errores = df[
    df["correcta"] == False
].copy()

errores = errores.sort_values(
    by="confianza_prediccion",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 20 ERRORES CON MAYOR CONFIANZA")
print("=" * 70)

for _, fila in errores.head(20).iterrows():

    print(
        f"\nArchivo: {fila['archivo']}"
    )

    print(
        f"Real: {fila['emocion_real']}"
    )

    print(
        f"Predicha: {fila['emocion_predicha']}"
    )

    print(
        f"Confianza: "
        f"{fila['confianza_prediccion'] * 100:.2f}%"
    )


# ============================================================
# 13. ENOJO MAL CLASIFICADO
# ============================================================

errores_enojo = df[
    (df["emocion_real"] == "Enojo") &
    (df["emocion_predicha"] != "Enojo")
].copy()

errores_enojo = errores_enojo.sort_values(
    by="confianza_prediccion",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 20 ENOJO MAL CLASIFICADO")
print("=" * 70)

for _, fila in errores_enojo.head(20).iterrows():

    print(
        f"\nArchivo: {fila['archivo']}"
    )

    print(
        f"Predicha: {fila['emocion_predicha']}"
    )

    print(
        f"Confianza: "
        f"{fila['confianza_prediccion'] * 100:.2f}%"
    )

    print(
        f"Prob. Enojo: "
        f"{fila['prob_Enojo'] * 100:.2f}%"
    )

    print(
        f"Prob. Felicidad: "
        f"{fila['prob_Felicidad'] * 100:.2f}%"
    )

    print(
        f"Prob. Neutral: "
        f"{fila['prob_Neutral'] * 100:.2f}%"
    )

    print(
        f"Prob. Tristeza: "
        f"{fila['prob_Tristeza'] * 100:.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISIS TERMINADO")
print("=" * 70)

print("\nArchivos generados:")
print(f"  ✅ {CSV_PATH}")

print("\nNO se modificó:")
print("  ❌ dataset/train")
print("  ❌ dataset/test")
print("  ❌ imágenes")
print("  ❌ modelo")

print("\nSiguiente paso:")
print("Revisar especialmente los errores de ENOJO.")
print("=" * 70)
