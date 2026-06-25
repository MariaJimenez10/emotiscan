from tensorflow.keras.models import load_model

modelo = load_model("modelo_emociones.h5")

print("✅ Modelo CNN cargado correctamente")

modelo.summary()