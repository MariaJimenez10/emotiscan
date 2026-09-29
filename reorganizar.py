import os
import shutil
import random

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Copia de seguridad
BACKUP_DIR = os.path.join(BASE_DIR, "dataset_backup")

# Clases
EMOCIONES = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

# Porcentaje para TEST
TEST_SIZE = 0.20

# Semilla para que la división sea reproducible
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ==========================================================
# VERIFICAR DATASET
# ==========================================================

print("=" * 60)
print("REORGANIZANDO DATASET DE EMOTISCAN")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    print("❌ No existe la carpeta dataset")
    exit()

if not os.path.exists(TRAIN_DIR):
    print("❌ No existe dataset/train")
    exit()

if not os.path.exists(TEST_DIR):
    print("❌ No existe dataset/test")
    exit()


# ==========================================================
# CREAR BACKUP
# ==========================================================

if os.path.exists(BACKUP_DIR):
    print("\n⚠️ La carpeta dataset_backup ya existe.")
    respuesta = input("¿Quieres eliminarla y crear un backup nuevo? (s/n): ")

    if respuesta.lower() == "s":
        shutil.rmtree(BACKUP_DIR)
    else:
        print("❌ Operación cancelada.")
        exit()

print("\n📦 Creando copia de seguridad...")

shutil.copytree(DATASET_DIR, BACKUP_DIR)

print(f"✅ Backup creado en:")
print(f"   {BACKUP_DIR}")


# ==========================================================
# RECOLECTAR TODAS LAS IMÁGENES
# ==========================================================

print("\n" + "=" * 60)
print("RECOLECTANDO IMÁGENES")
print("=" * 60)

extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

imagenes_por_clase = {}

for emocion in EMOCIONES:

    imagenes = []

    # Buscar en TRAIN
    carpeta_train = os.path.join(TRAIN_DIR, emocion)

    if os.path.exists(carpeta_train):

        for archivo in os.listdir(carpeta_train):

            ruta = os.path.join(carpeta_train, archivo)

            if os.path.isfile(ruta) and archivo.lower().endswith(extensiones):
                imagenes.append(ruta)

    # Buscar en TEST
    carpeta_test = os.path.join(TEST_DIR, emocion)

    if os.path.exists(carpeta_test):

        for archivo in os.listdir(carpeta_test):

            ruta = os.path.join(carpeta_test, archivo)

            if os.path.isfile(ruta) and archivo.lower().endswith(extensiones):
                imagenes.append(ruta)

    # Eliminar posibles duplicados por nombre/ruta
    imagenes = list(set(imagenes))

    random.shuffle(imagenes)

    imagenes_por_clase[emocion] = imagenes

    print(f"{emocion}: {len(imagenes)} imágenes")


# ==========================================================
# MOSTRAR DISTRIBUCIÓN OBJETIVO
# ==========================================================

print("\n" + "=" * 60)
print("DISTRIBUCIÓN OBJETIVO")
print("=" * 60)

for emocion in EMOCIONES:

    total = len(imagenes_por_clase[emocion])

    cantidad_test = round(total * TEST_SIZE)
    cantidad_train = total - cantidad_test

    print(
        f"{emocion}: "
        f"Train={cantidad_train} | "
        f"Test={cantidad_test} | "
        f"Total={total}"
    )


# ==========================================================
# CREAR CARPETAS TEMPORALES
# ==========================================================

TEMP_DIR = os.path.join(DATASET_DIR, "temp_reorganizacion")

if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR)

TEMP_TRAIN = os.path.join(TEMP_DIR, "train")
TEMP_TEST = os.path.join(TEMP_DIR, "test")

for emocion in EMOCIONES:

    os.makedirs(
        os.path.join(TEMP_TRAIN, emocion),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(TEMP_TEST, emocion),
        exist_ok=True
    )


# ==========================================================
# COPIAR IMÁGENES A LA NUEVA DISTRIBUCIÓN
# ==========================================================

print("\n" + "=" * 60)
print("REORGANIZANDO")
print("=" * 60)

for emocion in EMOCIONES:

    imagenes = imagenes_por_clase[emocion]

    total = len(imagenes)

    cantidad_test = round(total * TEST_SIZE)

    test_imagenes = imagenes[:cantidad_test]
    train_imagenes = imagenes[cantidad_test:]

    print(f"\n{emocion}")
    print(f"  Train: {len(train_imagenes)}")
    print(f"  Test:  {len(test_imagenes)}")

    # Copiar TRAIN
    for i, origen in enumerate(train_imagenes):

        extension = os.path.splitext(origen)[1]

        destino = os.path.join(
            TEMP_TRAIN,
            emocion,
            f"{emocion.lower()}_{i:05d}{extension}"
        )

        shutil.copy2(origen, destino)

    # Copiar TEST
    for i, origen in enumerate(test_imagenes):

        extension = os.path.splitext(origen)[1]

        destino = os.path.join(
            TEMP_TEST,
            emocion,
            f"{emocion.lower()}_{i:05d}{extension}"
        )

        shutil.copy2(origen, destino)


# ==========================================================
# ELIMINAR TRAIN Y TEST ANTIGUOS
# ==========================================================

print("\n" + "=" * 60)
print("REEMPLAZANDO DATASET")
print("=" * 60)

shutil.rmtree(TRAIN_DIR)
shutil.rmtree(TEST_DIR)


# ==========================================================
# MOVER NUEVAS CARPETAS
# ==========================================================

shutil.move(TEMP_TRAIN, TRAIN_DIR)
shutil.move(TEMP_TEST, TEST_DIR)

shutil.rmtree(TEMP_DIR)


# ==========================================================
# RESULTADO FINAL
# ==========================================================

print("\n" + "=" * 60)
print("✅ DATASET REORGANIZADO CORRECTAMENTE")
print("=" * 60)

print("\nTRAIN:")

total_train = 0

for emocion in EMOCIONES:

    carpeta = os.path.join(TRAIN_DIR, emocion)

    cantidad = len([
        f for f in os.listdir(carpeta)
        if f.lower().endswith(extensiones)
    ])

    total_train += cantidad

    print(f"{emocion}: {cantidad}")

print(f"TOTAL TRAIN: {total_train}")


print("\nTEST:")

total_test = 0

for emocion in EMOCIONES:

    carpeta = os.path.join(TEST_DIR, emocion)

    cantidad = len([
        f for f in os.listdir(carpeta)
        if f.lower().endswith(extensiones)
    ])

    total_test += cantidad

    print(f"{emocion}: {cantidad}")

print(f"TOTAL TEST: {total_test}")


print("\n" + "=" * 60)
print("BACKUP DISPONIBLE EN:")
print(BACKUP_DIR)
print("=" * 60)

print("\n🚀 Ahora puedes entrenar nuevamente tu ResNet50.")
