# diagnostico.py
import os
import cv2
import numpy as np

from import_os import (
    cargar_modelo, preparar_rostro,
    input_details, output_details,
    EMOCIONES, BASE_DIR
)

cargar_modelo()

print("\n" + "="*70)
print("DIAGNÓSTICO DE PREDICCIONES")
print("="*70)

# ⚠️ Pon aquí 4 imágenes REALES, una por emoción
# Si no tienes, usa la webcam para capturarlas ahora mismo
rutas = [
    ("enojo.jpg",   "Enojo"),
    ("feliz.jpg",   "Felicidad"),
    ("triste.jpg",  "Tristeza"),
    ("neutral.jpg", "Neutral"),
]

resultados = []

for archivo, esperado in rutas:
    ruta = os.path.join(BASE_DIR, archivo)
    if not os.path.exists(ruta):
        print(f"⚠️  No existe {archivo}, saltando...")
        continue

    img = cv2.imread(ruta)
    tensor = preparar_rostro(img)

    input_dtype = input_details[0]["dtype"]
    tensor = tensor.astype(input_dtype)

    interpreter = cargar_modelo()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    salida = interpreter.get_tensor(output_details[0]["index"])[0]

    print(f"\n📷 {archivo} (esperado: {esperado})")
    print(f"   RAW    : {salida}")
    print(f"   Suma   : {salida.sum():.4f}")
    print(f"   Min/Max: {salida.min():.4f} / {salida.max():.4f}")
    print(f"   Argmax : {EMOCIONES[np.argmax(salida)]}")

    dist = {e: round(float(v)*100, 2) for e, v in zip(EMOCIONES, salida)}
    print(f"   Dist   : {dist}")
    resultados.append((esperado, salida))

print("\n" + "="*70)
print("RESUMEN")
print("="*70)

if len(resultados) >= 2:
    primeras = [r[1] for r in resultados]
    iguales = all(np.allclose(primeras[0], p, atol=1e-3) for p in primeras)
    if iguales:
        print("❌ TODAS las entradas dan la MISMA salida")
        print("   → El modelo NO está usando la imagen. Revisa preprocess.")
    else:
        print("✅ Las salidas varían entre imágenes (el modelo sí reacciona)")

    # Ver si siempre gana la misma clase
    argmaxs = [EMOCIONES[np.argmax(r[1])] for r in resultados]
    if len(set(argmaxs)) == 1:
        print(f"❌ SIEMPRE gana '{argmaxs[0]}'")
        print("   → Sesgo del modelo hacia esa clase.")
        print("   → Causa: dataset desbalanceado o entrenamiento insuficiente.")
    else:
        print(f"✅ Varía el ganador: {argmaxs}")