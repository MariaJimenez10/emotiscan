import random
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MENSAJES_POR_EMOCION = {
    "Enojo": [
        "😡 Respira profundo, cuenta hasta 10",
        "🔥 La ira es pasajera, la paz es eterna",
        "💢 Canaliza tu energía en algo positivo",
        "⚡ La calma es tu superpoder",
        "😠 El enojo no resuelve nada",
        "👊 Da un paso atrás y respira",
        "🗯️ Expresa tus sentimientos con calma",
        "💥 Medita 5 minutos para calmarte",
        "⚠️ Esto también pasará",
        "🌋 La paciencia es la clave"
    ],
    "Felicidad": [
        "😊 ¡Qué hermoso momento! Disfrútalo",
        "✨ La felicidad se comparte",
        "🌟 Eres luz en la vida de otros",
        "🎉 Cada día es una oportunidad",
        "💫 Tu sonrisa ilumina el mundo",
        "🌈 La alegría está en los detalles",
        "🌸 La vida es bella",
        "☀️ Eres como el sol",
        "🎵 Música de corazón feliz",
        "❤️ El amor te acompaña"
    ],
    "Tristeza": [
        "😢 Está bien sentir tristeza",
        "🌧️ Después de la lluvia sale el sol",
        "🌫️ La tristeza es temporal",
        "🍂 Permítete sentir",
        "💔 El tiempo sana las heridas",
        "🌙 La noche pasa, el día llega",
        "🕯️ Busca apoyo en otros",
        "🌊 Las lágrimas limpian el alma",
        "🤍 Eres más fuerte de lo que crees",
        "⛅ El dolor es pasajero"
    ],
    "Sorpresa": [
        "😮 ¡El mundo está lleno de sorpresas!",
        "🤯 Lo inesperado trae oportunidades",
        "⭐ La vida te sorprende",
        "🎆 Disfruta el momento único",
        "✨ Las sorpresas son regalos",
        "🌠 Confía en el proceso",
        "🎁 Lo inesperado puede ser increíble",
        "🔮 Cada sorpresa trae una lección",
        "💥 Abre tu mente a lo nuevo",
        "🌈 Las mejores cosas llegan sin avisar"
    ],
    "Neutral": [
        "😐 Estás en equilibrio",
        "🧘 La calma es tu superpoder",
        "🌿 Disfruta el momento presente",
        "⚖️ El balance es la clave",
        "🌱 Cultiva la paz interior",
        "🍃 La tranquilidad es un tesoro",
        "💎 Valora estos momentos de paz",
        "🎯 Enfócate en lo que importa",
        "🌅 Cada día es una oportunidad",
        "✨ La paz interior es tu riqueza"
    ]
}

_mensajes_usados = defaultdict(list)

def obtener_mensaje(emocion):
    """Obtiene mensaje aleatorio sin repetir"""
    emocion_normalizada = None
    for key in MENSAJES_POR_EMOCION:
        if emocion.lower() == key.lower():
            emocion_normalizada = key
            break
    
    if emocion_normalizada is None:
        emocion_normalizada = "Neutral"
    
    mensajes_disponibles = MENSAJES_POR_EMOCION.get(emocion_normalizada, [])
    
    if not mensajes_disponibles:
        return "Cuida de ti mismo 💪"
    
    usados = _mensajes_usados.get(emocion_normalizada, [])
    disponibles = [msg for msg in mensajes_disponibles if msg not in usados]
    
    if not disponibles:
        logger.info(f"🔄 Reiniciando mensajes para {emocion_normalizada}")
        _mensajes_usados[emocion_normalizada] = []
        disponibles = mensajes_disponibles
    
    mensaje = random.choice(disponibles)
    _mensajes_usados[emocion_normalizada].append(mensaje)
    
    return mensaje

def mostrar_estado_mensajes():
    """Muestra estado actual"""
    for emocion, mensajes in MENSAJES_POR_EMOCION.items():
        usados = len(_mensajes_usados.get(emocion, []))
        total = len(mensajes)
        logger.info(f"  {emocion}: {usados}/{total} usados")