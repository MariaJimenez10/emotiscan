import random
from collections import defaultdict

# 10 mensajes ÚNICOS por cada emoción (SIN REPETICIÓN)
MENSAJES_POR_EMOCION = {
    "Enojo": [
        "😡 Respira profundo, cuenta hasta 10 y piensa en algo positivo",
        "🔥 La ira es pasajera, la paz es eterna, tómate un momento",
        "💢 Canaliza tu energía en hacer ejercicio o escribir",
        "⚡ La calma es tu superpoder, úsala ahora",
        "😠 El enojo no resuelve nada, la comunicación sí",
        "👊 Da un paso atrás y mira la situación con claridad",
        "🗯️ Expresa tus sentimientos de forma asertiva",
        "💥 La meditación de 5 minutos puede cambiar tu día",
        "⚠️ Recuerda: esto también pasará, todo es temporal",
        "🌋 La paciencia es la clave para superar la ira"
    ],
    "Felicidad": [
        "😊 ¡Qué hermoso momento! Disfrútalo al máximo",
        "✨ La felicidad se multiplica cuando la compartes",
        "🌟 Eres luz en la vida de quienes te rodean",
        "🎉 Cada día es una nueva oportunidad para ser feliz",
        "💫 Tu sonrisa es contagiosa, ¡sigue así!",
        "🌈 La alegría está en los pequeños detalles de la vida",
        "🌸 La vida es bella cuando la miras con optimismo",
        "☀️ Eres como el sol, iluminas todo a tu paso",
        "🎵 La música de tu corazón suena a felicidad",
        "❤️ El amor y la alegría te acompañan siempre"
    ],
    "Tristeza": [
        "😢 Está bien sentir tristeza, es parte de ser humano",
        "🌧️ Después de la tormenta siempre sale el sol",
        "🌫️ La tristeza es temporal, la esperanza es eterna",
        "🍂 Permítete sentir, pero no te quedes atrapado",
        "💔 El tiempo y el apoyo sanan las heridas",
        "🌙 La noche es oscura, pero el amanecer siempre llega",
        "🕯️ Busca apoyo en quienes te quieren y te entienden",
        "🌊 Las lágrimas limpian el alma y renuevan el espíritu",
        "🤍 Eres más fuerte de lo que crees, confía en ti",
        "⛅ El dolor es pasajero, tu fuerza interior es permanente"
    ],
    "Sorpresa": [
        "😮 ¡El mundo está lleno de sorpresas maravillosas!",
        "🤯 Lo inesperado trae nuevas oportunidades y aprendizajes",
        "⭐ La vida te sorprende cuando menos lo esperas",
        "🎆 Disfruta el momento, es único e irrepetible",
        "✨ Las sorpresas son regalos que la vida te da",
        "🌠 Confía en el proceso, todo tiene su propósito",
        "🎁 Lo inesperado puede ser el inicio de algo increíble",
        "🔮 Cada sorpresa trae una lección valiosa",
        "💥 Abre tu mente a las nuevas posibilidades",
        "🌈 Las mejores cosas llegan sin avisar, ¡disfrútalas!"
    ],
    "Neutral": [
        "😐 Estás en equilibrio, en paz contigo mismo",
        "🧘 La calma es tu superpoder, cultívala",
        "🌿 Disfruta del momento presente, es un regalo",
        "⚖️ El balance es la clave para una vida plena",
        "🌱 Cultiva la paz interior día a día",
        "🍃 La tranquilidad es un tesoro que mereces",
        "💎 Valora estos momentos de paz y serenidad",
        "🎯 Enfócate en lo que realmente importa en tu vida",
        "🌅 Cada día es una nueva oportunidad para ser feliz",
        "✨ La paz interior es la mayor riqueza que puedes tener"
    ]
}

# Control de mensajes usados (global)
_mensajes_usados = defaultdict(list)

def obtener_mensaje(emocion):
    """
    Obtiene un mensaje aleatorio SIN REPETIR hasta que se usen los 10
    
    Args:
        emocion (str): La emoción para la cual obtener mensaje
    
    Returns:
        str: Mensaje aleatorio no repetido
    """
    # Normalizar emoción (asegurar que existe en el diccionario)
    emocion_normalizada = None
    
    # Buscar coincidencia exacta o parcial
    for key in MENSAJES_POR_EMOCION:
        if emocion.lower() == key.lower():
            emocion_normalizada = key
            break
    
    # Si no se encuentra, buscar coincidencia parcial
    if emocion_normalizada is None:
        for key in MENSAJES_POR_EMOCION:
            if emocion.lower() in key.lower() or key.lower() in emocion.lower():
                emocion_normalizada = key
                break
    
    # Si aún no se encuentra, usar Neutral
    if emocion_normalizada is None:
        print(f"⚠️ Emoción '{emocion}' no encontrada, usando 'Neutral'")
        emocion_normalizada = "Neutral"
    
    # Obtener lista de mensajes disponibles para esta emoción
    mensajes_disponibles = MENSAJES_POR_EMOCION.get(emocion_normalizada, [])
    
    # Si no hay mensajes, retornar un mensaje por defecto
    if not mensajes_disponibles:
        return "Cuida de ti mismo y sigue adelante 💪"
    
    # Obtener mensajes ya usados
    usados = _mensajes_usados.get(emocion_normalizada, [])
    
    # Filtrar mensajes no usados
    disponibles = [msg for msg in mensajes_disponibles if msg not in usados]
    
    # Si ya se usaron todos los mensajes, REINICIAR el contador
    if not disponibles:
        print(f"🔄 Reiniciando mensajes para {emocion_normalizada} (ya se usaron los 10)")
        _mensajes_usados[emocion_normalizada] = []
        disponibles = mensajes_disponibles
    
    # Seleccionar un mensaje aleatorio
    mensaje = random.choice(disponibles)
    
    # Marcar como usado
    _mensajes_usados[emocion_normalizada].append(mensaje)
    
    # Mostrar cuántos quedan disponibles
    restantes = len(disponibles) - 1
    if restantes > 0:
        print(f"📝 {emocion_normalizada}: {restantes} mensajes disponibles")
    else:
        print(f"🔄 {emocion_normalizada}: Último mensaje usado, se reiniciará")
    
    return mensaje

def obtener_todos_mensajes():
    """Retorna todos los mensajes organizados por emoción"""
    return MENSAJES_POR_EMOCION

def reiniciar_contadores():
    """Reinicia el contador de mensajes usados"""
    global _mensajes_usados
    _mensajes_usados = defaultdict(list)
    print("🔄 Contadores de mensajes reiniciados")

def mostrar_estado_mensajes():
    """Muestra el estado actual de los mensajes usados"""
    print("\n📊 ESTADO DE MENSAJES:")
    for emocion, mensajes in MENSAJES_POR_EMOCION.items():
        usados = len(_mensajes_usados.get(emocion, []))
        total = len(mensajes)
        print(f"  {emocion}: {usados}/{total} usados")
    print("")

# ============================================
# PRUEBA RÁPIDA (10 mensajes por emoción sin repetir)
# ============================================
if __name__ == "__main__":
    print("🧪 Probando mensajes (10 por emoción sin repetición)...\n")
    
    # Probar cada emoción
    for emocion in MENSAJES_POR_EMOCION.keys():
        print(f"\n📌 {emocion}:")
        mensajes_usados = []
        
        # Obtener 10 mensajes (deberían ser todos diferentes)
        for i in range(10):
            mensaje = obtener_mensaje(emocion)
            mensajes_usados.append(mensaje)
            print(f"  {i+1}. {mensaje}")
        
        # Verificar que no haya repeticiones
        if len(set(mensajes_usados)) == 10:
            print(f"  ✅ 10 mensajes ÚNICOS para {emocion}")
        else:
            print(f"  ⚠️ Hay repeticiones en {emocion}")
    
    print("\n✅ Prueba completada")