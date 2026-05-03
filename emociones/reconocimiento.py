import cv2
import mediapipe as mp
from deepface import DeepFace
from collections import Counter

# ------------------------------
# ABRIR CAMARA
# ------------------------------

cap = cv2.VideoCapture(0)

historial = []

# ------------------------------
# MEDIAPIPE
# ------------------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

mp_draw = mp.solutions.drawing_utils

# ------------------------------
# CONSEJOS
# ------------------------------

def consejo_emocion(emocion):

    consejos = {

        "Felicidad": "Aprovecha tu energia para hacer algo que te guste.",
        "Tristeza": "Escuchar musica o hablar con alguien puede ayudarte.",
        "Enojo": "Respira profundo y tomate un momento para relajarte.",
        "Neutral": "Es un buen momento para concentrarte en tus tareas."

    }

    return consejos.get(emocion, "")

# ------------------------------
# BOTON SALIR
# ------------------------------

boton_salir = (450, 20, 620, 70)

def click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if boton_salir[0] < x < boton_salir[2] and boton_salir[1] < y < boton_salir[3]:

            cap.release()
            cv2.destroyAllWindows()
            exit()

cv2.namedWindow("IA Emocional")
cv2.setMouseCallback("IA Emocional", click)

# ------------------------------
# BUCLE PRINCIPAL
# ------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    # ------------------------------
    # ESCANER AZUL
    # ------------------------------

    cv2.rectangle(frame, (150, 80), (500, 400), (255, 0, 0), 2)

    cv2.putText(frame,
                "SCANNING FACE...",
                (200, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    # ------------------------------
    # PUNTOS DEL ROSTRO
    # ------------------------------

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=1)
            )

    # ------------------------------
    # DETECTAR EMOCION
    # ------------------------------

    try:

        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emociones = result[0]['emotion']

        seleccion = {

            'angry': emociones['angry'],
            'happy': emociones['happy'],
            'sad': emociones['sad'],
            'neutral': emociones['neutral']

        }

        emocion_en = max(seleccion, key=seleccion.get)

        historial.append(emocion_en)

        if len(historial) > 10:
            historial.pop(0)

        emocion_estable = Counter(historial).most_common(1)[0][0]

        traduccion = {

            'angry': 'Enojo',
            'happy': 'Felicidad',
            'sad': 'Tristeza',
            'neutral': 'Neutral'

        }

        emocion_es = traduccion[emocion_estable]

        consejo = consejo_emocion(emocion_es)

        # ------------------------------
        # MOSTRAR EMOCION
        # ------------------------------

        cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), -1)

        cv2.putText(frame,
                    f"Emocion: {emocion_es}",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        # ------------------------------
        # CONSEJO
        # ------------------------------

        cv2.rectangle(frame, (20, 100), (620, 170), (0, 0, 0), -1)

        cv2.putText(frame,
                    consejo,
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

    except:
        pass

    # ------------------------------
    # BOTON SALIR
    # ------------------------------

    cv2.rectangle(frame,
                  (boton_salir[0], boton_salir[1]),
                  (boton_salir[2], boton_salir[3]),
                  (0, 0, 255),
                  -1)

    cv2.putText(frame,
                "SALIR",
                (500, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    # ------------------------------
    # MOSTRAR CAMARA
    # ------------------------------

    cv2.imshow("IA Emocional", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()