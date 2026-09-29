import os
import cv2

from predict import cargar_modelo, predecir


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_IMAGEN = os.path.join(
    BASE_DIR,
    "debug_rostros",
    "rostro_detectado.jpg"
)


# ==========================================================
# INICIO
# ==========================================================

print("=" * 60)
print("🧪 PRUEBA DEL RECORTE DE LA CÁMARA")
print("=" * 60)

print(f"📁 Imagen:")
print(f"   {RUTA_IMAGEN}")


# ==========================================================
# COMPROBAR IMAGEN
# ==========================================================

if not os.path.exists(RUTA_IMAGEN):
    raise FileNotFoundError(
        f"No existe la imagen:\n{RUTA_IMAGEN}"
    )

imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    raise ValueError(
        f"No se pudo abrir la imagen:\n{RUTA_IMAGEN}"
    )

print(
    f"📷 Tamaño: {imagen.shape}"
)


# ==========================================================
# CARGAR MODELO
# ==========================================================

print()
print("=" * 60)
print("🧠 CARGANDO MODELO")
print("=" * 60)

cargar_modelo()


# ==========================================================
# PREDICCIÓN
# ==========================================================

print()
print("=" * 60)
print("🔮 REALIZANDO PREDICCIÓN")
print("=" * 60)

emocion, confianza, todas = predecir(
    imagen
)


# ==========================================================
# RESULTADO
# ==========================================================

print()
print("=" * 60)
print("🎭 RESULTADO")
print("=" * 60)

print(
    f"Emoción: {emocion}"
)

print(
    f"Confianza: {confianza:.2f}%"
)

print()
print("📊 TODAS LAS EMOCIONES:")

for nombre, porcentaje in todas.items():

    print(
        f"{nombre:<12}: {porcentaje:>6.2f}%"
    )

print("=" * 60)
print("✅ PRUEBA TERMINADA")
print("=" * 60)