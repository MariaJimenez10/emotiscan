from pathlib import Path

ruta = Path("templates/index.html")
contenido = ruta.read_text(encoding="utf-8")

marcador = '''    <a href="/inicio" style="display:block;margin-top:20px">'''

panel = '''    <section id="estadisticasSesion" style="display:none; margin-top:24px; padding:20px; background:#f4f6fb; color:#20283a; border-radius:12px; text-align:left;">
        <h3>Resumen de la sesión automática</h3>

        <p>Total de detecciones: <strong id="sesionTotal">0</strong></p>
        <p>Emoción principal: <strong id="sesionEmocionPrincipal">Sin detecciones</strong></p>
        <p>Duración: <strong id="sesionDuracion">0 s</strong></p>
        <p>Confianza promedio: <strong id="sesionConfianza">0%</strong></p>
        <p>Promedio de detecciones por minuto: <strong id="sesionPromedioMinuto">0</strong></p>

        <h4>Emociones por minuto</h4>
        <div id="sesionPorMinuto">Todavía no hay detecciones registradas.</div>

        <h4>Distribución acumulada</h4>
        <div id="sesionDistribucion">Todavía no hay estadísticas.</div>
    </section>

'''

if "id=\"estadisticasSesion\"" in contenido:
    print("El panel ya existe. No se realizaron cambios.")
elif marcador not in contenido:
    print("No se encontró el punto de inserción. No se realizaron cambios.")
else:
    contenido = contenido.replace(marcador, panel + marcador, 1)
    ruta.write_text(contenido, encoding="utf-8")
    print("Panel de estadísticas agregado correctamente.")
