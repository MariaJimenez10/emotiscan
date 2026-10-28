from pathlib import Path

ruta = Path("templates/index.html")
contenido = ruta.read_text(encoding="utf-8")

# Evitar aplicar el cambio dos veces.
if "async function actualizarEstadisticasSesion(" in contenido:
    print("La función de estadísticas ya existe. No se realizaron cambios.")
    raise SystemExit(0)

reemplazos = [
    (
        'let autoTimer = null;\nlet sessionId = null;',
        'let autoTimer = null;\nlet statsTimer = null;\nlet sessionId = null;'
    ),
    (
        '/* MODO AUTOMATICO */',
        r'''/* ESTADISTICAS DE LA SESION */

async function actualizarEstadisticasSesion(id = sessionId) {
    if (!id) return;

    try {
        const response = await fetch(
            `/sesion/estadisticas?sesion_id=${encodeURIComponent(id)}`,
            { credentials: "same-origin" }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "No se pudieron consultar las estadísticas.");
        }

        document.getElementById("estadisticasSesion").style.display = "block";
        document.getElementById("sesionTotal").textContent = data.total_detecciones ?? 0;
        document.getElementById("sesionEmocionPrincipal").textContent =
            data.emocion_principal || "Sin detecciones";

        const segundos = Number(data.duracion_segundos) || 0;
        const minutos = Math.floor(segundos / 60);
        const resto = segundos % 60;

        document.getElementById("sesionDuracion").textContent =
            `${minutos} min ${resto} s`;

        document.getElementById("sesionConfianza").textContent =
            `${Number(data.confianza_promedio || 0).toFixed(2)}%`;

        document.getElementById("sesionPromedioMinuto").textContent =
            Number(data.promedio_detecciones_minuto || 0).toFixed(2);

        const distribucion = document.getElementById("sesionDistribucion");
        distribucion.replaceChildren();

        if (Array.isArray(data.emociones) && data.emociones.length) {
            data.emociones.forEach(item => {
                const fila = document.createElement("p");
                fila.textContent =
                    `${item.emocion}: ${item.cantidad} detecciones (${item.porcentaje}%)`;
                distribucion.appendChild(fila);
            });
        } else {
            distribucion.textContent = "Todavía no hay detecciones.";
        }

        const porMinuto = document.getElementById("sesionPorMinuto");
        porMinuto.replaceChildren();

        if (Array.isArray(data.por_minuto) && data.por_minuto.length) {
            data.por_minuto.forEach(item => {
                const fila = document.createElement("p");
                const detalle = Object.entries(item.emociones || {})
                    .map(([emocion, cantidad]) => `${emocion}: ${cantidad}`)
                    .join(", ");

                fila.textContent =
                    `Minuto ${item.minuto} (${item.total} detecciones): ${detalle}`;
                porMinuto.appendChild(fila);
            });
        } else {
            porMinuto.textContent = "Todavía no hay detecciones registradas.";
        }
    } catch (error) {
        console.error("Error al actualizar estadísticas:", error);
    }
}

/* MODO AUTOMATICO */'''
    ),
    (
        '        setStatus("Modo automático activo.");\n\n        captureAndAnalyze();',
        '''        setStatus("Modo automático activo.");
        document.getElementById("estadisticasSesion").style.display = "block";

        actualizarEstadisticasSesion(sessionId);

        if (statsTimer !== null) clearInterval(statsTimer);
        statsTimer = setInterval(() => {
            if (autoMode && sessionId !== null) {
                actualizarEstadisticasSesion(sessionId);
            }
        }, 5000);

        captureAndAnalyze();'''
    ),
    (
        '''    if (autoTimer !== null) {
        clearInterval(autoTimer);
        autoTimer = null;
    }

    autoBtn.textContent = "Modo automático";''',
        '''    if (autoTimer !== null) {
        clearInterval(autoTimer);
        autoTimer = null;
    }

    if (statsTimer !== null) {
        clearInterval(statsTimer);
        statsTimer = null;
    }

    autoBtn.textContent = "Modo automático";'''
    ),
    (
        '''    }
}
function stopTracks() {''',
        '''    }

    // Consultar una última vez para mostrar el resumen final.
    if (idToFinalize !== null) {
        await actualizarEstadisticasSesion(idToFinalize);
    }
}
function stopTracks() {'''
    )
]

# Validar todos los puntos antes de escribir para evitar cambios parciales.
for antiguo, nuevo in reemplazos:
    cantidad = contenido.count(antiguo)
    if cantidad != 1:
        print("No se realizó ningún cambio.")
        print(f"Se esperaba encontrar una vez este fragmento, pero se encontró {cantidad}:")
        print(antiguo[:180])
        raise SystemExit(1)

respaldo = ruta.with_name("index.html.antes_conectar_estadisticas.bak")
if not respaldo.exists():
    respaldo.write_text(contenido, encoding="utf-8")

for antiguo, nuevo in reemplazos:
    contenido = contenido.replace(antiguo, nuevo, 1)

ruta.write_text(contenido, encoding="utf-8")
print("Estadísticas conectadas. Copia de seguridad creada:", respaldo)
