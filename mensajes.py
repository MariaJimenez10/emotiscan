import random

mensajes = {

"Feliz":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Triste":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Enojado":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Neutral":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Miedo":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Disgusto":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
],

"Sorprendido":[
"Mensaje 1",
"Mensaje 2",
"Mensaje 3",
"Mensaje 4",
"Mensaje 5",
"Mensaje 6",
"Mensaje 7",
"Mensaje 8",
"Mensaje 9",
"Mensaje 10"
]

}

historial = {}

def obtener_mensaje(emocion):

    if emocion not in historial:
        historial[emocion] = []

    disponibles = list(
        set(range(len(mensajes[emocion])))
        -
        set(historial[emocion])
    )

    if len(disponibles) == 0:

        historial[emocion] = []

        disponibles = list(range(len(mensajes[emocion])))

    indice = random.choice(disponibles)

    historial[emocion].append(indice)

    return mensajes[emocion][indice]