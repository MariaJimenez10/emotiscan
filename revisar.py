
from pathlib import Path
from collections import Counter
import hashlib

BASE_DIR = Path(__file__).resolve().parent
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]


def obtener_imagenes(carpeta):
    if not carpeta.exists():
        return []

    return sorted(
        archivo
        for archivo in carpeta.rglob("*")
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES
    )


def hash_archivo(ruta):
    sha = hashlib.sha256()

    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)

    return sha.hexdigest()


def revisar_conjunto(nombre):
    carpeta = BASE_DIR / "dataset" / nombre

    print(f"\n{'=' * 50}")
    print(f"CONJUNTO: dataset/{nombre}")
    print(f"{'=' * 50}")

    if not carpeta.exists():
        print("La carpeta no existe.")
        return {}

    hashes = {}

    for clase in CLASES:
        carpeta_clase = carpeta / clase
        imagenes = obtener_imagenes(carpeta_clase)

        print(f"{clase}: {len(imagenes)} imágenes")

        for imagen in imagenes:
            try:
                digest = hash_archivo(imagen)
                hashes.setdefault(digest, []).append(
                    (clase, imagen)
                )
            except OSError as error:
                print(f"No se pudo leer {imagen}: {error}")

    return hashes


def main():
    train = revisar_conjunto("train")
    test = revisar_conjunto("test")

    carpeta_limpio = BASE_DIR / "dataset" / "train_limpio"

    print(f"\n{'=' * 50}")
    print("CONJUNTO: dataset/train_limpio")
    print(f"{'=' * 50}")

    if carpeta_limpio.exists():
        for clase in CLASES:
            imagenes = obtener_imagenes(carpeta_limpio / clase)
            print(f"{clase}: {len(imagenes)} imágenes")
    else:
        print("La carpeta train_limpio no existe.")

    print(f"\n{'=' * 50}")
    print("IMÁGENES IDÉNTICAS ENTRE TRAIN Y TEST")
    print(f"{'=' * 50}")

    comunes = set(train.keys()) & set(test.keys())

    if not comunes:
        print("No se encontraron archivos idénticos por contenido.")
    else:
        print(f"Contenidos idénticos encontrados: {len(comunes)}")

        for digest in sorted(comunes):
            print(f"\nSHA256: {digest[:16]}...")

            for clase, ruta in train[digest]:
                print(f"  TRAIN [{clase}]: {ruta.relative_to(BASE_DIR)}")

            for clase, ruta in test[digest]:
                print(f"  TEST  [{clase}]: {ruta.relative_to(BASE_DIR)}")

    print("\nRevisión terminada. No se modificó ningún archivo.")


if __name__ == "__main__":
    main()
