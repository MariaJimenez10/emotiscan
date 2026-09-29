import os
import hashlib

TRAIN = "dataset/train"
TEST = "dataset/test"


def obtener_hash(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()


def obtener_imagenes(carpeta):
    hashes = {}

    for emocion in os.listdir(carpeta):
        ruta_emocion = os.path.join(carpeta, emocion)

        if not os.path.isdir(ruta_emocion):
            continue

        for archivo in os.listdir(ruta_emocion):
            ruta = os.path.join(ruta_emocion, archivo)

            if os.path.isfile(ruta):
                try:
                    hash_imagen = obtener_hash(ruta)
                    hashes[hash_imagen] = ruta
                except Exception as e:
                    print(f"Error leyendo {ruta}: {e}")

    return hashes


print("=" * 60)
print("COMPROBANDO DUPLICADOS ENTRE TRAIN Y TEST")
print("=" * 60)

train_hashes = obtener_imagenes(TRAIN)
test_hashes = obtener_imagenes(TEST)

duplicados = set(train_hashes.keys()) & set(test_hashes.keys())

print()
print(f"Imágenes en TRAIN: {len(train_hashes)}")
print(f"Imágenes en TEST:  {len(test_hashes)}")
print(f"Duplicados:        {len(duplicados)}")
print()

if duplicados:
    print("⚠️ SE ENCONTRARON IMÁGENES REPETIDAS:")
    
    for hash_imagen in duplicados:
        print()
        print("TRAIN:")
        print(train_hashes[hash_imagen])
        print("TEST:")
        print(test_hashes[hash_imagen])
else:
    print("✅ NO HAY IMÁGENES DUPLICADAS ENTRE TRAIN Y TEST.")