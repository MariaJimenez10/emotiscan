from deepface import DeepFace
import cv2
import numpy as np

def detectar_rostro(imagen):

    try:

        faces = DeepFace.extract_faces(
            img_path=imagen,
            detector_backend='retinaface',
            enforce_detection=False,
            align=True
        )

        if len(faces) == 0:
            return None

        rostro = faces[0]["face"]

        rostro = (rostro * 255).astype(np.uint8)

        rostro = cv2.cvtColor(rostro, cv2.COLOR_RGB2BGR)

        return rostro

    except Exception as e:
        print(e)
        return None