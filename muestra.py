
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random

random.seed(123)
clases = ["Enojo", "Felicidad", "Neutral", "Tristeza"]
extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
miniaturas = []

for clase in clases:
    carpeta = Path("dataset/test") / clase
    archivos = [
        p for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() in extensiones
    ]

    seleccion = random.sample(archivos, min(12, len(archivos)))

    for archivo in seleccion:
        try:
            with Image.open(archivo) as imagen:
                imagen = ImageOps.exif_transpose(imagen).convert("RGB")
                imagen.thumbnail((140, 140))
                miniaturas.append((clase, archivo.name, imagen.copy()))
        except Exception:
            pass

ancho = 4 * 190
alto = ((len(miniaturas) + 3) // 4) * 180
lamina = Image.new("RGB", (ancho, alto), "white")
dibujo = ImageDraw.Draw(lamina)

for i, (clase, nombre, imagen) in enumerate(miniaturas):
    x = (i % 4) * 190
    y = (i // 4) * 180
    lamina.paste(imagen, (x, y + 22))
    dibujo.text((x + 3, y + 3), clase, fill="black")
    dibujo.text((x + 3, y + 165), nombre[:27], fill="black")

salida = Path("muestra_test_revision.jpg")
lamina.save(salida, quality=92)

print(f"Lamina creada: {salida.resolve()}")
print(f"Imagenes incluidas: {len(miniaturas)}")
