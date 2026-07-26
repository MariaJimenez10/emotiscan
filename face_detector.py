from deepface import DeepFace
import cv2
import numpy as np

def detectar_rostro(imagen):
    try:
        print("Entró a detectar_rostro")

        faces = DeepFace.extract_faces(
            img_path=imagen,
            detector_backend="opencv",
            enforce_detection=False,
            align=False
        )

        print("Faces encontradas:", len(faces))

        if not faces:
            print("No se encontraron rostros")
            return None

        rostro = faces[0].get("face")

        if rostro is None:
            print("La clave 'face' es None")
            return None

        print("Shape del rostro:", rostro.shape)
        print("Tipo:", rostro.dtype)

        if rostro.dtype != np.uint8:
            rostro = (rostro * 255).clip(0, 255).astype(np.uint8)

        if len(rostro.shape) == 3 and rostro.shape[2] == 3:
            rostro = cv2.cvtColor(rostro, cv2.COLOR_RGB2BGR)

        return rostro

    except Exception as e:
        print("Error en detectar_rostro:", e)
        return None