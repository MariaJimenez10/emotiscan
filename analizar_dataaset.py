import os
from collections import Counter, defaultdict
from PIL import Image
import hashlib

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test")

EMOCIONES = ["Enojo", "Felicidad", "Neutral", "Tristeza"]

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# FUNCIONES
# ============================================================

def obtener_imagenes(carpeta):
    """
    Devuelve todas las rutas de imágenes dentro de una carpeta.
    """
    imagenes = []

    if not os.path.exists(carpeta):
        return imagenes

    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)

        if os.path.isfile(ruta):
            extension = os.path.splitext(archivo)[1].lower()

            if extension in EXTENSIONES:
                imagenes.append(ruta)

    return imagenes


def contar_por_clase(base_path):
    """
    Cuenta imágenes por emoción.
    """
    resultados = {}

    for emocion in EMOCIONES:
        carpeta = os.path.join(base_path, emocion)
        imagenes = obtener_imagenes(carpeta)
        resultados[emocion] = len(imagenes)

    return resultados


def verificar_imagen(ruta):
    """
    Comprueba si una imagen puede abrirse correctamente.
    También obtiene tamaño y modo de color.
    """
    try:
        with Image.open(ruta) as img:
            img.verify()

        # Volvemos a abrir porque verify() invalida el objeto.
        with Image.open(ruta) as img:
            ancho, alto = img.size
            modo = img.mode

        return True, ancho, alto, modo, ""

    except Exception as e:
        return False, None, None, None, str(e)


def calcular_hash(ruta):
    """
    Calcula un hash MD5 de la imagen.
    Sirve para detectar imágenes exactamente iguales.
    """
    try:
        hash_md5 = hashlib.md5()

        with open(ruta, "rb") as archivo:
            for bloque in iter(lambda: archivo.read(8192), b""):
                hash_md5.update(bloque)

        return hash_md5.hexdigest()

    except Exception:
        return None


def analizar_conjunto(nombre, base_path):
    """
    Analiza todas las imágenes de train o test.
    """
    print("\n" + "=" * 70)
    print(f"ANALIZANDO {nombre}")
    print("=" * 70)

    total = 0
    corruptas = []
    extensiones = Counter()
    dimensiones = Counter()
    modos = Counter()
    hashes = defaultdict(list)

    for emocion in EMOCIONES:

        carpeta = os.path.join(base_path, emocion)

        print(f"\n📁 {emocion}")

        if not os.path.exists(carpeta):
            print("   ❌ CARPETA NO ENCONTRADA")
            continue

        imagenes = obtener_imagenes(carpeta)

        print(f"   Imágenes: {len(imagenes)}")

        total += len(imagenes)

        for ruta in imagenes:

            extension = os.path.splitext(ruta)[1].lower()
            extensiones[extension] += 1

            correcta, ancho, alto, modo, error = verificar_imagen(ruta)

            if not correcta:
                corruptas.append((ruta, error))
                continue

            dimensiones[(ancho, alto)] += 1
            modos[modo] += 1

            hash_imagen = calcular_hash(ruta)

            if hash_imagen:
                hashes[hash_imagen].append(ruta)

    print("\n" + "-" * 70)
    print(f"TOTAL {nombre}: {total}")

    print("\n📐 DIMENSIONES MÁS FRECUENTES:")

    for (ancho, alto), cantidad in dimensiones.most_common(10):
        porcentaje = cantidad / total * 100 if total else 0

        print(
            f"   {ancho} x {alto}: "
            f"{cantidad} imágenes ({porcentaje:.2f}%)"
        )

    print("\n🎨 MODOS DE COLOR:")

    for modo, cantidad in modos.most_common():
        porcentaje = cantidad / total * 100 if total else 0

        print(
            f"   {modo}: "
            f"{cantidad} imágenes ({porcentaje:.2f}%)"
        )

    print("\n📄 EXTENSIONES:")

    for extension, cantidad in extensiones.most_common():
        print(f"   {extension}: {cantidad}")

    print("\n🚨 IMÁGENES CORRUPTAS:")

    if corruptas:
        print(f"   ❌ Encontradas: {len(corruptas)}")

        for ruta, error in corruptas[:20]:
            print(f"\n   {ruta}")
            print(f"   Error: {error}")

        if len(corruptas) > 20:
            print(
                f"\n   ... y {len(corruptas) - 20} imágenes más."
            )

    else:
        print("   ✅ No se encontraron imágenes corruptas.")

    duplicados = {
        hash_: rutas
        for hash_, rutas in hashes.items()
        if len(rutas) > 1
    }

    print("\n🔁 DUPLICADOS DENTRO DE ESTE CONJUNTO:")

    if duplicados:
        print(f"   ⚠️ Grupos duplicados: {len(duplicados)}")

        mostrados = 0

        for hash_, rutas in duplicados.items():

            print("\n   Grupo duplicado:")

            for ruta in rutas:
                print(f"      {ruta}")

            mostrados += 1

            if mostrados >= 10:
                break

        if len(duplicados) > 10:
            print(
                f"\n   ... y {len(duplicados) - 10} grupos más."
            )

    else:
        print("   ✅ No se encontraron duplicados internos.")

    return {
        "total": total,
        "corruptas": corruptas,
        "hashes": hashes
    }


# ============================================================
# INICIO
# ============================================================

print("=" * 70)
print("ANÁLISIS DEL DATASET EMOTISCAN")
print("=" * 70)

print(f"\n📂 TRAIN: {TRAIN_PATH}")
print(f"📂 TEST : {TEST_PATH}")

# ============================================================
# COMPROBAR CARPETAS
# ============================================================

print("\n" + "=" * 70)
print("COMPROBANDO ESTRUCTURA")
print("=" * 70)

for nombre, ruta in [
    ("TRAIN", TRAIN_PATH),
    ("TEST", TEST_PATH)
]:

    if os.path.exists(ruta):
        print(f"✅ {nombre}: {ruta}")
    else:
        print(f"❌ {nombre} NO EXISTE: {ruta}")

# ============================================================
# CANTIDAD POR CLASE
# ============================================================

print("\n" + "=" * 70)
print("CANTIDAD DE IMÁGENES POR EMOCIÓN")
print("=" * 70)

train_counts = contar_por_clase(TRAIN_PATH)
test_counts = contar_por_clase(TEST_PATH)

print("\n📁 TRAIN")

for emocion in EMOCIONES:
    print(f"   {emocion}: {train_counts[emocion]}")

total_train = sum(train_counts.values())

print(f"   TOTAL: {total_train}")

print("\n📁 TEST")

for emocion in EMOCIONES:
    print(f"   {emocion}: {test_counts[emocion]}")

total_test = sum(test_counts.values())

print(f"   TOTAL: {total_test}")

print("\n📊 TOTAL DATASET")

for emocion in EMOCIONES:
    total_emocion = (
        train_counts[emocion] +
        test_counts[emocion]
    )

    print(f"   {emocion}: {total_emocion}")

print(f"\n   TOTAL GENERAL: {total_train + total_test}")

# ============================================================
# PORCENTAJES
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUCIÓN DEL DATASET")
print("=" * 70)

print("\nTRAIN:")

for emocion in EMOCIONES:

    porcentaje = (
        train_counts[emocion] / total_train * 100
        if total_train else 0
    )

    print(
        f"   {emocion}: "
        f"{train_counts[emocion]} "
        f"({porcentaje:.2f}%)"
    )

print("\nTEST:")

for emocion in EMOCIONES:

    porcentaje = (
        test_counts[emocion] / total_test * 100
        if total_test else 0
    )

    print(
        f"   {emocion}: "
        f"{test_counts[emocion]} "
        f"({porcentaje:.2f}%)"
    )

# ============================================================
# ANALIZAR TRAIN
# ============================================================

train_info = analizar_conjunto(
    "TRAIN",
    TRAIN_PATH
)

# ============================================================
# ANALIZAR TEST
# ============================================================

test_info = analizar_conjunto(
    "TEST",
    TEST_PATH
)

# ============================================================
# COMPARAR TRAIN VS TEST
# ============================================================

print("\n" + "=" * 70)
print("COMPARACIÓN TRAIN VS TEST")
print("=" * 70)

print(
    f"\n{'EMOCIÓN':<15}"
    f"{'TRAIN':>10}"
    f"{'TEST':>10}"
    f"{'TOTAL':>10}"
)

print("-" * 50)

for emocion in EMOCIONES:

    train = train_counts[emocion]
    test = test_counts[emocion]
    total = train + test

    print(
        f"{emocion:<15}"
        f"{train:>10}"
        f"{test:>10}"
        f"{total:>10}"
    )

print("-" * 50)

print(
    f"{'TOTAL':<15}"
    f"{total_train:>10}"
    f"{total_test:>10}"
    f"{total_train + total_test:>10}"
)

# ============================================================
# COMPROBAR DUPLICADOS ENTRE TRAIN Y TEST
# ============================================================

print("\n" + "=" * 70)
print("COMPROBANDO DUPLICADOS ENTRE TRAIN Y TEST")
print("=" * 70)

train_hashes = {}
test_hashes = {}

print("\nCalculando hashes de TRAIN...")

for emocion in EMOCIONES:

    carpeta = os.path.join(TRAIN_PATH, emocion)

    for ruta in obtener_imagenes(carpeta):

        hash_imagen = calcular_hash(ruta)

        if hash_imagen:
            train_hashes[hash_imagen] = ruta

print(f"Hashes TRAIN: {len(train_hashes)}")

print("\nCalculando hashes de TEST...")

for emocion in EMOCIONES:

    carpeta = os.path.join(TEST_PATH, emocion)

    for ruta in obtener_imagenes(carpeta):

        hash_imagen = calcular_hash(ruta)

        if hash_imagen:
            test_hashes[hash_imagen] = ruta

print(f"Hashes TEST: {len(test_hashes)}")

duplicados_train_test = (
    set(train_hashes.keys())
    & set(test_hashes.keys())
)

print(
    f"\nDuplicados exactos TRAIN/TEST: "
    f"{len(duplicados_train_test)}"
)

if duplicados_train_test:

    print("\n⚠️ SE ENCONTRARON DUPLICADOS:")

    for hash_imagen in list(duplicados_train_test)[:20]:

        print("\nTRAIN:")
        print(f"   {train_hashes[hash_imagen]}")

        print("TEST:")
        print(f"   {test_hashes[hash_imagen]}")

else:

    print(
        "✅ NO HAY IMÁGENES DUPLICADAS "
        "ENTRE TRAIN Y TEST."
    )

# ============================================================
# COMPROBAR NOMBRES REPETIDOS ENTRE TRAIN Y TEST
# ============================================================

print("\n" + "=" * 70)
print("COMPROBANDO NOMBRES REPETIDOS ENTRE TRAIN Y TEST")
print("=" * 70)

train_names = defaultdict(list)
test_names = defaultdict(list)

for emocion in EMOCIONES:

    carpeta = os.path.join(TRAIN_PATH, emocion)

    for ruta in obtener_imagenes(carpeta):

        nombre = os.path.basename(ruta).lower()
        train_names[nombre].append(ruta)

for emocion in EMOCIONES:

    carpeta = os.path.join(TEST_PATH, emocion)

    for ruta in obtener_imagenes(carpeta):

        nombre = os.path.basename(ruta).lower()
        test_names[nombre].append(ruta)

nombres_repetidos = (
    set(train_names.keys())
    & set(test_names.keys())
)

print(
    f"\nNombres de archivo repetidos: "
    f"{len(nombres_repetidos)}"
)

if nombres_repetidos:

    print("\n⚠️ NOMBRES REPETIDOS:")

    for nombre in list(nombres_repetidos)[:20]:

        print(f"\n   {nombre}")

        print("   TRAIN:")

        for ruta in train_names[nombre]:
            print(f"      {ruta}")

        print("   TEST:")

        for ruta in test_names[nombre]:
            print(f"      {ruta}")

else:

    print(
        "✅ No hay nombres de archivo repetidos "
        "entre TRAIN y TEST."
    )

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)

print(f"\n📊 TRAIN: {total_train}")
print(f"📊 TEST : {total_test}")
print(f"📊 TOTAL: {total_train + total_test}")

print("\n📁 TRAIN:")

for emocion in EMOCIONES:
    print(f"   {emocion}: {train_counts[emocion]}")

print("\n📁 TEST:")

for emocion in EMOCIONES:
    print(f"   {emocion}: {test_counts[emocion]}")

print(
    f"\n🚨 Imágenes corruptas TRAIN: "
    f"{len(train_info['corruptas'])}"
)

print(
    f"🚨 Imágenes corruptas TEST: "
    f"{len(test_info['corruptas'])}"
)

print(
    f"\n🔁 Duplicados exactos TRAIN/TEST: "
    f"{len(duplicados_train_test)}"
)

print(
    f"🔁 Nombres repetidos TRAIN/TEST: "
    f"{len(nombres_repetidos)}"
)

print("\n" + "=" * 70)
print("ANÁLISIS TERMINADO")
print("=" * 70)

print(
    "\n✅ Este programa NO modificó ninguna imagen "
    "ni carpeta del dataset."
)

print(
    "\n👉 Copia TODO el resultado de esta ejecución "
    "y envíamelo."
)
