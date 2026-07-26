from deepface import DeepFace
import cv2
import numpy as np

def detectar_rostro(imagen):
    try:
        faces = DeepFace.extract_faces(
            img_path=imagen,
            detector_backend="opencv",
            enforce_detection=False,
            align=False
        )

        # Si no encuentra rostros
        if not faces:
            return None

        # Obtener el rostro
        rostro = faces[0].get("face")

        if rostro is None:
            return None

        # Convertir a uint8 solo si está normalizado (0-1)
        if rostro.dtype != np.uint8:
            rostro = (rostro * 255).clip(0, 255).astype(np.uint8)

        # Convertir de RGB a BGR para OpenCV
        if len(rostro.shape) == 3 and rostro.shape[2] == 3:
            rostro = cv2.cvtColor(rostro, cv2.COLOR_RGB2BGR)

        return rostro

    except Exception as e:
        print(f"Error en detectar_rostro: {e}")
        return None