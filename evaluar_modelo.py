
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from predict import EMOCIONES, predecir

# Reducir los mensajes de cada prediccion.
logging.disable(logging.CRITICAL)

BASE_DIR = Path(__file__).resolve().parent
CARPETA_TEST = BASE_DIR / "dataset" / "test"
CLASES = list(EMOCIONES)
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not CARPETA_TEST.is_dir():
        print(f"No existe la carpeta: {CARPETA_TEST}")
        return

    matriz = np.zeros(
        (len(CLASES), len(CLASES)),
        dtype=np.int64
    )

    errores = []
    fallidas = 0
    total = 0

    indices = {clase: i for i, clase in enumerate(CLASES)}

    print("Evaluando el modelo actual...")
    print("No se entrenara ni reemplazara ningun modelo.\n")

    for real in CLASES:
        carpeta = CARPETA_TEST / real

        if not carpeta.is_dir():
            print(f"Falta la carpeta: {carpeta}")
            return

        archivos = sorted(
            p for p in carpeta.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        )

        correctas = 0

        for archivo in archivos:
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                fallidas += 1
                continue

            try:
                predicha, confianza, todas, consejo = predecir(imagen)
            except Exception as error:
                fallidas += 1
                print(f"Error en {archivo.name}: {error}")
                continue

            if predicha not in indices:
                fallidas += 1
                continue

            matriz[indices[real], indices[predicha]] += 1
            total += 1

            if predicha == real:
                correctas += 1
            else:
                errores.append(
                    (real, predicha, archivo.name, confianza)
                )

        print(
            f"{real}: {correctas}/{len(archivos)} "
            "imagenes clasificadas correctamente "
            "(el total procesado puede ser menor si hay errores de lectura)"
        )

    print("\n========== MATRIZ DE CONFUSION ==========")
    print("Filas = emocion real; columnas = emocion predicha")
    print("Orden:", CLASES)
    print(matriz)

    aciertos = int(np.trace(matriz))
    exactitud = 100 * aciertos / total if total else 0

    print("\n========== RESULTADOS ==========")
    print(f"Imagenes evaluadas: {total}")
    print(f"Imagenes no procesadas: {fallidas}")
    print(f"Aciertos: {aciertos}")
    print(f"Exactitud global: {exactitud:.2f}%")

    print("\n========== METRICAS POR EMOCION ==========")

    for i, clase in enumerate(CLASES):
        verdaderos = int(matriz[i, i])
        total_real = int(matriz[i, :].sum())
        total_predicho = int(matriz[:, i].sum())

        recall = (
            100 * verdaderos / total_real
            if total_real else 0
        )
        precision = (
            100 * verdaderos / total_predicho
            if total_predicho else 0
        )

        print(
            f"{clase}: recall={recall:.2f}% | "
            f"precision={precision:.2f}% | "
            f"casos reales={total_real}"
        )

    print("\n========== PRIMEROS 30 ERRORES ==========")
    for real, predicha, nombre, confianza in errores[:30]:
        print(
            f"Real: {real} | Predicha: {predicha} | "
            f"{confianza:.2f}% | {nombre}"
        )

    print("\nEvaluacion terminada. No se modifico el modelo.")


if __name__ == "__main__":
    main()
