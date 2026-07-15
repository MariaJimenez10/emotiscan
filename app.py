import os
# 🔥 CONFIGURACIÓN DE MEMORIA - AL PRINCIPIO DE TODO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import tensorflow as tf
# 🔥 DESACTIVAR GPU Y LIMITAR THREADS
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
import sqlite3
from tensorflow.keras.applications.resnet50 import preprocess_input
import gc
import sys

# 🔥 REDUCIR USO DE MEMORIA DE OPENCV
cv2.setNumThreads(0)

# -----------------------------
# APP FLASK
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "emocionesIA_segura_2026")

# -----------------------------
# BASE DE DATOS
# -----------------------------
DATABASE = '/tmp/emotiscan.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT UNIQUE,
            password TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emociones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT,
            emocion TEXT,
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Base de datos SQLite inicializada")

init_db()

# -----------------------------
# DETECTOR DE ROSTRO
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# CARGAR MODELO RESNET50 - UNA SOLA VEZ
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_resnet50_emociones.h5")

print("🔄 Cargando modelo ResNet50...")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    
    # 🔥 CARGAR CON COMPILE=FALSE PARA AHORRAR MEMORIA
    modelo_cnn = load_model(MODEL_PATH, compile=False)
    print("✅ Modelo ResNet50 cargado correctamente")
    print(f"📊 Entrada: {modelo_cnn.input_shape}")
    print(f"📊 Salida: {modelo_cnn.output_shape}")
    
    num_salidas = modelo_cnn.output_shape[-1]
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    sys.exit(1)

# =============================
# EMOCIONES - DETECCIÓN AUTOMÁTICA
# =============================

# 🔥 DETECTAR NÚMERO DE EMOCIONES DEL MODELO
if num_salidas == 7:
    EMOCIONES = ["Enojo", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
elif num_salidas == 4:
    EMOCIONES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
elif num_salidas == 3:
    EMOCIONES = ["Enojo", "Felicidad", "Neutral"]
else:
    EMOCIONES = [f"Emoción_{i}" for i in range(num_salidas)]

print(f"📊 Emociones: {EMOCIONES}")

CONSEJOS = {
    "Enojo": "😡 Respira profundamente y cuenta hasta 10.",
    "Asco": "🤢 Busca algo positivo en tu entorno.",
    "Miedo": "😨 Enfócate en lo que puedes controlar.",
    "Felicidad": "😊 ¡Qué bien! Disfruta este momento.",
    "Tristeza": "😢 Habla con alguien de confianza.",
    "Sorpresa": "😮 Tómate un momento para procesarlo.",
    "Neutral": "😐 Estás en equilibrio."
}

# =============================
# FUNCIÓN DE PREDICCIÓN - ULTRA OPTIMIZADA
# =============================

def predecir_cnn(img):
    """Predicción optimizada al máximo para ahorrar memoria"""
    try:
        # 🔥 REDUCIR IMAGEN AL MÍNIMO
        if img.shape[0] > 300 or img.shape[1] > 300:
            img = cv2.resize(img, (300, 300))
        
        # 🔥 DETECCIÓN DE ROSTRO RÁPIDA
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Reducir resolución para detección más rápida y menos memoria
        gray_small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        
        rostros = face_detector.detectMultiScale(
            gray_small,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            x, y = x*2, y*2
            w, h = w*2, h*2
            x, y = max(0, x), max(0, y)
            img = img[y:y+h, x:x+w]
        
        # 🔥 PROCESAR EN 224x224
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # 🔥 PREDECIR CON BATCH_SIZE=1
        pred = modelo_cnn.predict(img, verbose=0, batch_size=1)
        
        indice = np.argmax(pred[0])
        emocion = EMOCIONES[indice] if indice < len(EMOCIONES) else EMOCIONES[0]
        
        # 🔥 LIMPIAR MEMORIA
        del img, pred, gray, gray_small
        if 'rostros' in locals():
            del rostros
        gc.collect()
        
        return emocion
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        gc.collect()
        return "Neutral"

# =============================
# RUTAS
# =============================

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def validar():
    usuario = request.form.get("usuario")
    password = request.form.get("password")
    
    if not usuario or not password:
        return "❌ Usuario y contraseña son requeridos", 400
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM usuarios WHERE usuario = ?", (usuario,))
    result = cursor.fetchone()
    conn.close()

    if result and check_password_hash(result[0], password):
        session["user"] = usuario 
        return redirect("/inicio")
    else:
        return """
        <script>
            alert('❌ Usuario o contraseña incorrectos');
            window.location.href = '/';
        </script>
        """

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/guardar", methods=["POST"])
def guardar_usuario():
    usuario = request.form.get("usuario")
    password = request.form.get("password")
    
    if not usuario or not password:
        return "❌ Usuario y contraseña son requeridos", 400
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO usuarios (usuario, password) VALUES (?, ?)",
            (usuario, hashed_password)
        )
        conn.commit()
        conn.close()
        return redirect("/")
    except sqlite3.IntegrityError:
        return "⚠️ El usuario ya existe. <a href='/register'>Intentar de nuevo</a>"
    except Exception as e:
        print(f"Error en registro: {e}")
        return "❌ Error interno del servidor", 500

@app.route("/inicio")
def inicio():
    if "user" not in session:
        return redirect("/")
    
    ahora = datetime.now()
    return render_template(
        "index.html",
        usuario=session["user"],
        fecha=ahora.strftime("%Y-%m-%d"),
        hora=ahora.strftime("%H:%M:%S")
    )

@app.route("/analizar", methods=["POST"])
def analizar():
    if "user" not in session:
        return jsonify({"error": "No hay sesión activa"}), 401
    
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400
        
        # Decodificar imagen
        image_data = data["image"].split(";base64,")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400
        
        emocion = predecir_cnn(img)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")
        
        # Guardar en BD
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO emociones (usuario, emocion) VALUES (?, ?)",
            (session["user"], emocion)
        )
        conn.commit()
        conn.close()
        
        # 🔥 LIMPIAR MEMORIA
        gc.collect()
        
        return jsonify({
            "emotion": emocion,
            "advice": consejo
        })
        
    except Exception as e:
        print(f"❌ ERROR ANALIZAR: {e}")
        gc.collect()
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT emocion, COUNT(*)
        FROM emociones
        WHERE usuario = ?
        GROUP BY emocion
    """, (session["user"],))
    
    datos = cursor.fetchall()
    conn.close()
    
    conteo = {emocion: 0 for emocion in EMOCIONES}
    for row in datos:
        if row["emocion"] in conteo:
            conteo[row["emocion"]] = row[1]
    
    return render_template("dashboard.html", conteo=conteo)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route('/imagen')
def imagen():
    return render_template('imagen.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'imagen' not in request.files:
            return jsonify({'estado': 'error', 'detalle': 'No se envió imagen'}), 400
        
        archivo = request.files['imagen']
        if archivo.filename == "":
            return jsonify({"estado": "error", "detalle": "No se seleccionó imagen"}), 400
        
        file_bytes = np.frombuffer(archivo.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'estado': 'error', 'detalle': 'Error al leer imagen'}), 400
        
        emocion = predecir_cnn(img)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")
        
        if "user" in session:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion) VALUES (?, ?)",
                (session["user"], emocion)
            )
            conn.commit()
            conn.close()
        
        gc.collect()
        
        return jsonify({
            'emocion': emocion,
            'consejo': consejo,
            'estado': 'success'
        })
        
    except Exception as e:
        print(f"❌ ERROR predict_image: {e}")
        gc.collect()
        return jsonify({'estado': 'error', 'detalle': str(e)}), 500

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "EmotiScan funcionando"}), 200

@app.route('/ping')
def ping():
    return "pong", 200

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"❌ Error global: {e}")
    return jsonify({"error": str(e), "status": "error"}), 500

# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Servidor iniciado en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)