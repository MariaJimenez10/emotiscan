import os
# 🔥 CONFIGURACIÓN DE MEMORIA - AL PRINCIPIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
# 🔥 CONFIGURACIÓN EXTREMA DE MEMORIA
tf.config.set_visible_devices([], 'GPU')  # Desactivar GPU completamente
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Limitar memoria de CPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('CPU')[0], True)
except:
    pass

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

# -----------------------------
# APP FLASK
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "emocionesIA")

# -----------------------------
# CONEXIÓN SQLITE
# -----------------------------
DATABASE = '/tmp/emotiscan.db'

def get_db():
    """Obtener conexión a la base de datos"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializar tablas"""
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
    print("✅ Base de datos SQLite inicializada en:", DATABASE)

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
    print("📊 Entrada:", modelo_cnn.input_shape)
    print("📊 Salida:", modelo_cnn.output_shape)
    
    # Verificar número de salidas
    num_salidas = modelo_cnn.output_shape[-1]
    print(f"📊 Número de emociones: {num_salidas}")
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("🔄 Creando modelo de respaldo...")
    
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)   # 🔥 Menos neuronas para ahorrar memoria
    x = Dense(32, activation='relu')(x)   # 🔥 Menos neuronas
    predictions = Dense(4, activation='softmax')(x)  # 4 emociones
    modelo_cnn = Model(inputs=base_model.input, outputs=predictions)
    print("✅ Modelo de respaldo creado (optimizado)")

# =============================
# FUNCION CNN - OPTIMIZADA
# =============================

# 🔥 DETECTAR AUTOMÁTICAMENTE EL NÚMERO DE EMOCIONES DEL MODELO
NUM_EMOCIONES = modelo_cnn.output_shape[-1]

# Lista de emociones según el modelo
if NUM_EMOCIONES == 7:
    EMOCIONES = ["Enojo", "Asco", "Miedo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
elif NUM_EMOCIONES == 4:
    EMOCIONES = ["Enojo", "Felicidad", "Tristeza", "Neutral"]
elif NUM_EMOCIONES == 3:
    EMOCIONES = ["Enojo", "Felicidad", "Neutral"]
else:
    # Si el modelo tiene otro número, usar genéricos
    EMOCIONES = [f"Emoción_{i}" for i in range(NUM_EMOCIONES)]

print(f"📊 Usando {len(EMOCIONES)} emociones: {EMOCIONES}")

# Diccionario de consejos
CONSEJOS = {
    "Enojo": "😡 Respira profundamente y cuenta hasta 10. Tómate un momento para relajarte.",
    "Asco": "🤢 Intenta encontrar algo positivo en tu entorno.",
    "Miedo": "😨 Recuerda que es normal sentir miedo. Respira y enfócate en lo que puedes controlar.",
    "Felicidad": "😊 ¡Qué bien! Disfruta este momento y compártelo con alguien especial.",
    "Tristeza": "😢 Está bien sentir tristeza. Habla con alguien de confianza.",
    "Sorpresa": "😮 ¡Qué sorpresa! Tómate un momento para procesarlo.",
    "Neutral": "😐 Te ves tranquilo y enfocado. Sigue así, estás en equilibrio."
}

def predecir_cnn(img):
    """Predice la emoción con optimización de memoria"""
    try:
        # 🔥 REDUCIR TAMAÑO DE IMAGEN
        if img.shape[0] > 300 or img.shape[1] > 300:
            img = cv2.resize(img, (300, 300))
        
        # Detectar rostro
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        
        # Procesar para el modelo
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # 🔥 PREDECIR
        pred = modelo_cnn.predict(img, verbose=0, batch_size=1)
        
        indice = np.argmax(pred[0])
        confianza = np.max(pred[0])
        
        # Asegurar que el índice sea válido
        if indice >= len(EMOCIONES):
            indice = 0  # Fallback a la primera emoción
        
        emocion = EMOCIONES[indice]
        
        print(f"📊 Predicción: {emocion} ({confianza:.2%})")
        
        # 🔥 LIMPIAR MEMORIA
        del img, pred, gray, gray_small
        gc.collect()
        
        return emocion
        
    except Exception as e:
        print(f"❌ Error en predecir_cnn: {e}")
        gc.collect()
        return "Neutral"

def consejo_emocion(emocion):
    """Obtiene un consejo según la emoción"""
    return CONSEJOS.get(emocion, "🔍 No fue posible determinar la emoción. ¡Cuida de ti mismo!")

# -----------------------------
# RUTAS
# -----------------------------

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/login", methods=["GET"])
def login_get():
    return redirect("/")

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
        
        image_data = data["image"].split(";base64,")[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400
        
        emocion = predecir_cnn(img)
        consejo = consejo_emocion(emocion)
        
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
        print("❌ ERROR ANALIZAR:", str(e))
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
        emocion = row["emocion"]
        cantidad = row[1]
        if emocion in conteo:
            conteo[emocion] = cantidad
    
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
            return jsonify({
                'estado': 'error',
                'detalle': 'No se envió imagen'
            }), 400
        
        archivo = request.files['imagen']
        
        if archivo.filename == "":
            return jsonify({
                "estado": "error",
                "detalle": "No se seleccionó ninguna imagen"
            }), 400
        
        file_bytes = np.frombuffer(archivo.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'estado': 'error',
                'detalle': 'Error al leer imagen'
            }), 400
        
        emocion = predecir_cnn(img)
        consejo = consejo_emocion(emocion)
        
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
        print("❌ ERROR predict_image:", str(e))
        gc.collect()
        return jsonify({
            'emocion': 'No se pudo analizar la imagen',
            'detalle': str(e),
            'estado': 'error'
        }), 500

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
    return jsonify({
        "error": str(e),
        "status": "error"
    }), 500

# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Iniciando servidor en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)