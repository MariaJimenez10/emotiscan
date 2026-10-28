
from pathlib import Path
import csv
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, ImageDraw
from tensorflow.keras.applications.resnet50 import preprocess_input

MODELO = Path("modelo/resnet50_emociones_limpio.keras")
CARPETA = Path("dataset_limpio/test/Enojo")
SALIDA = Path("revision_furia_predicciones.jpg")
CSV_SALIDA = Path("revision_furia_predicciones.csv")

CLASES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not MODELO.exists():
    raise FileNotFoundError(f"No existe el modelo: {MODELO}")

if not CARPETA.exists():
    raise FileNotFoundError(f"No existe la carpeta: {CARPETA}")

modelo = tf.keras.models.load_model(MODELO)

archivos = sorted(
    p for p in CARPETA.iterdir()
    if p.is_file() and p.suffix.lower() in EXTENSIONES
)

resultados = []

for ruta in archivos:
    with Image.open(ruta) as original:
        imagen = original.convert("RGB").resize((224, 224))
        arreglo = np.asarray(imagen, dtype=np.float32)
        arreglo = np.expand_dims(arreglo, axis=0)
        arreglo = preprocess_input(arreglo)

    probabilidades = modelo.predict(arreglo, verbose=0)[0]
    prediccion = int(np.argmax(probabilidades))

    resultados.append({
        "ruta": ruta,
        "prediccion": CLASES[prediccion],
        "probabilidades": probabilidades.copy(),
        "enojo": float(probabilidades[0])
    })

# Primero las imágenes a las que el modelo asigna menos Enojo.
resultados.sort(key=lambda x: x["enojo"])

with CSV_SALIDA.open("w", newline="", encoding="utf-8-sig") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([
        "imagen", "prediccion", "Enojo",
        "Felicidad", "Tristeza", "Neutral"
    ])

    for r in resultados:
        p = r["probabilidades"]
        escritor.writerow([
            r["ruta"].name,
            r["prediccion"],
            *[round(float(p[i]) * 100, 2) for i in range(4)]
        ])

seleccion = resultados[:20]
ancho = 260
alto_imagen = 200
alto_texto = 75
columnas = 4
filas = (len(seleccion) + columnas - 1) // columnas

lamina = Image.new(
    "RGB",
    (columnas * ancho, filas * (alto_imagen + alto_texto)),
    "white"
)
dibujo = ImageDraw.Draw(lamina)

for i, r in enumerate(seleccion):
    x = (i % columnas) * ancho
    y = (i // columnas) * (alto_imagen + alto_texto)

    with Image.open(r["ruta"]) as original:
        imagen = ImageOps.contain(
            original.convert("RGB"),
            (ancho - 16, alto_imagen - 12)
        )

    px = x + (ancho - imagen.width) // 2
    py = y + (alto_imagen - imagen.height) // 2
    lamina.paste(imagen, (px, py))

    p = r["probabilidades"]
    dibujo.text((x + 5, y + alto_imagen), r["ruta"].name[:36], fill="black")
    dibujo.text(
        (x + 5, y + alto_imagen + 20),
        f"Enojo: {p[0]*100:.1f}% | Pred.: {r['prediccion']}",
        fill="black"
    )
    dibujo.text(
        (x + 5, y + alto_imagen + 40),
        f"Felicidad {p[1]*100:.0f}%  Tristeza {p[2]*100:.0f}%",
        fill="black"
    )

lamina.save(SALIDA, quality=95)

aciertos = sum(r["prediccion"] == "Enojo" for r in resultados)

print(f"Imágenes analizadas: {len(resultados)}")
print(f"Predichas como Enojo: {aciertos}")
print(f"Porcentaje: {aciertos / len(resultados) * 100:.2f}%")
print(f"Lámina: {SALIDA.resolve()}")
print(f"Resultados CSV: {CSV_SALIDA.resolve()}")
