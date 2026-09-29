import os
import hashlib
import shutil

TRAIN = "dataset/train"
TEST = "dataset/test"
DESTINO = "dataset/duplicados_train"


def obtener_hash(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()


def obtener_hashes(carpeta):
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
print("ELIMINANDO DUPLICADOS DE TRAIN")
print("=" * 60)

# Crear carpeta de respaldo
os.makedirs(DESTINO, exist_ok=True)

print("\nAnalizando TEST...")
test_hashes = obtener_hashes(TEST)

print("Analizando TRAIN...")
train_hashes = obtener_hashes(TRAIN)

duplicados = set(train_hashes.keys()) & set(test_hashes.keys())

print(f"\nDuplicados encontrados: {len(duplicados)}")

movidas = 0

for hash_imagen in duplicados:

    ruta_train = train_hashes[hash_imagen]

    emocion = os.path.basename(os.path.dirname(ruta_train))
    archivo = os.path.basename(ruta_train)

    carpeta_destino = os.path.join(DESTINO, emocion)
    os.makedirs(carpeta_destino, exist_ok=True)

    destino = os.path.join(carpeta_destino, archivo)

    shutil.move(ruta_train, destino)

    print(f"\nMovida:")
    print(f"  TRAIN: {ruta_train}")
    print(f"  --> {destino}")

    movidas += 1


print("\n" + "=" * 60)
print(f"TOTAL DE IMÁGENES MOVIDAS: {movidas}")
print("=" * 60)

if movidas == len(duplicados):
    print("✅ Todos los duplicados fueron retirados de TRAIN.")
else:
    print("⚠️ Revisar el resultado.")