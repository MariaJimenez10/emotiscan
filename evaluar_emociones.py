
import logging
from pathlib import Path
from collections import Counter

import cv2

from predict import predecir, EMOCIONES

logging.disable(logging.CRITICAL)

BASE_DIR = Path(__file__).resolve().parent
CARPETA_TEST = BASE_DIR / "dataset" / "test"

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIMITE_POR_EMOCION = 30


def main():
    if not CARPETA_TEST.exists():
        print(f"No existe la carpeta: {CARPETA_TEST}")
        return

    # Matriz: emoción real frente a emoción predicha.
    matriz = {
        real: {predicha: 0 for predicha in EMOCIONES}
        for real in EMOCIONES
    }

    predicciones_por_emocion = Counter()
    probabilidades_por_emocion = {
        real: {emocion: 0.0 for emocion in EMOCIONES}
        for real in EMOCIONES
    }

    total_por_emocion = Counter()
    errores = 0

    print("\n" + "=" * 65)
    print("EVALUACIÓN DE LAS CUATRO EMOCIONES")
    print("=" * 65)

    for emocion_real in EMOCIONES:
        carpeta = CARPETA_TEST / emocion_real

        if not carpeta.exists():
            print(f"\nATENCIÓN: No existe la carpeta {carpeta}")
            continue

        archivos = sorted(
            archivo for archivo in carpeta.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES
        )[:LIMITE_POR_EMOCION]

        if not archivos:
            print(f"\nNo hay imágenes en {emocion_real}")
            continue

        print(
            f"\nEvaluando {emocion_real}: "
            f"{len(archivos)} imágenes"
        )

        for archivo in archivos:
            imagen = cv2.imread(str(archivo))

            if imagen is None:
                print(f"No se pudo leer: {archivo.name}")
                errores += 1
                continue

            try:
                emocion_predicha, confianza, todas, consejo = (
                    predecir(imagen)
                )

                # Compatibilidad si alguna parte de la aplicación
                # muestra "Furia" en lugar de "Enojo".
                if emocion_predicha == "Furia":
                    emocion_predicha = "Enojo"

                if emocion_predicha not in EMOCIONES:
                    print(
                        f"Etiqueta desconocida en {archivo.name}: "
                        f"{emocion_predicha}"
                    )
                    errores += 1
                    continue

                matriz[emocion_real][emocion_predicha] += 1
                predicciones_por_emocion[emocion_predicha] += 1
                total_por_emocion[emocion_real] += 1

                for emocion in EMOCIONES:
                    probabilidades_por_emocion[emocion_real][emocion] += (
                        float(todas[emocion])
                    )

                print(
                    f"Real: {emocion_real:10} | "
                    f"Predicción: {emocion_predicha:10} | "
                    f"Confianza: {confianza:.2f}%"
                )

            except Exception as error:
                print(f"Error en {archivo.name}: {error}")
                errores += 1

    total = sum(total_por_emocion.values())
    aciertos = sum(
        matriz[emocion][emocion]
        for emocion in EMOCIONES
    )

    print("\n" + "=" * 65)
    print("RESULTADOS GENERALES")
    print("=" * 65)

    print(f"Imágenes evaluadas correctamente: {total}")
    print(f"Errores de lectura o predicción: {errores}")

    if total == 0:
        print(
            "\nNo se evaluaron imágenes. "
            "Revisa las carpetas de dataset/test."
        )
        return

    print(
        f"Exactitud global: {aciertos}/{total} "
        f"({aciertos / total * 100:.2f}%)"
    )

    print("\n" + "=" * 65)
    print("RESULTADOS POR EMOCIÓN")
    print("=" * 65)

    for emocion in EMOCIONES:
        cantidad_real = total_por_emocion[emocion]
        verdaderos_positivos = matriz[emocion][emocion]
        cantidad_predicha = predicciones_por_emocion[emocion]

        recall = (
            verdaderos_positivos / cantidad_real * 100
            if cantidad_real else 0
        )

        precision = (
            verdaderos_positivos / cantidad_predicha * 100
            if cantidad_predicha else 0
        )

        print(f"\n{emocion}")
        print(
            f"  Imágenes reales evaluadas: {cantidad_real}"
        )
        print(
            f"  Reconocidas correctamente: "
            f"{verdaderos_positivos}"
        )
        print(f"  Recall: {recall:.2f}%")
        print(f"  Precisión: {precision:.2f}%")

    print("\n" + "=" * 65)
    print("MATRIZ DE CONFUSIÓN")
    print("Filas = emoción real | Columnas = predicción")
    print("=" * 65)

    encabezado = f"{'Real / Pred.':15}"
    for emocion in EMOCIONES:
        encabezado += f"{emocion:>12}"
    print(encabezado)

    for real in EMOCIONES:
        fila = f"{real:15}"
        for predicha in EMOCIONES:
            fila += f"{matriz[real][predicha]:12}"
        print(fila)

    print("\n" + "=" * 65)
    print("PROMEDIO DE PROBABILIDADES POR EMOCIÓN REAL")
    print("=" * 65)

    for real in EMOCIONES:
        cantidad = total_por_emocion[real]

        if cantidad == 0:
            continue

        print(f"\nImágenes reales de {real}:")

        for emocion in EMOCIONES:
            promedio = (
                probabilidades_por_emocion[real][emocion] / cantidad
            )
            print(f"  {emocion}: {promedio:.2f}%")

    print("\nEvaluación terminada.")
    print("No se modificaron los modelos ni el entrenamiento.")


if __name__ == "__main__":
    main()
