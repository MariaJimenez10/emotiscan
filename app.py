import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# Limitar threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Configurar para usar menos memoria
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=128)]
            )
    except RuntimeError as e:
        print(e)
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
import sqlite3
from tensorflow.keras.applications.resnet50 import preprocess_input
# -----------------------------
# CONFIGURACIÓN DE TENSORFLOW PARA MEMORIA
# -----------------------------
import tensorflow as tf

# 🔥 LIMITAR USO DE MEMORIA
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 🔥 LIMITAR HILOS DE CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 🔥 CONFIGURAR PARA AHORRAR MEMORIA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


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
    
    # Tabla usuarios
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    # Tabla emociones
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

# Inicializar BD
init_db()

# -----------------------------
# DETECTOR DE ROSTRO
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# CARGAR MODELO RESNET50 - VERSIÓN OPTIMIZADA
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelo_resnet50_emociones.h5"
)

# 🔥 CARGAR CON COMPILE=FALSE PARA AHORRAR MEMORIA
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    
    modelo_cnn = load_model(MODEL_PATH, compile=False)  # 👈 Compile=False ahorra memoria
    print("✅ Modelo ResNet50 cargado correctamente")
    print("📊 Entrada:", modelo_cnn.input_shape)
    print("📊 Salida:", modelo_cnn.output_shape)
    
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    # Crear modelo de respaldo (más ligero)
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)  # 🔥 Menos neuronas
    x = Dense(64, activation='relu')(x)   # 🔥 Menos neuronas
    predictions = Dense(7, activation='softmax')(x)
    modelo_cnn = Model(inputs=base_model.input, outputs=predictions)
    print("✅ Modelo de respaldo creado (más ligero)")


# Manejo de errores en carga de modelo
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    
    modelo_cnn = load_model(MODEL_PATH)
    print("✅ Modelo ResNet50 cargado correctamente")
    print("📊 Entrada:", modelo_cnn.input_shape)
    print("📊 Salida:", modelo_cnn.output_shape)
    
    # Verificar que el modelo tenga las 4 emociones
    if modelo_cnn.output_shape[-1] != 4:
        print(f"⚠️ El modelo tiene {modelo_cnn.output_shape[-1]} salidas, se esperaban 4")
        
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("⚠️ Usando modelo de respaldo...")
    # Crear modelo de respaldo
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 emociones
    modelo_cnn = Model(inputs=base_model.input, outputs=predictions)
    modelo_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Modelo de respaldo creado")

# =============================
# FUNCION CNN - CORREGIDA
# =============================

# Lista completa de 3 emociones
EMOCIONES = [
    "Enojo",       # 0
    "Felicidad",   # 1
    "Tristeza",    # 2
    "Neutral"      # 3
]

# Diccionario de consejos para 4 emociones
CONSEJOS = {
    "Enojo": "😡 Respira profundamente y cuenta hasta 10. Tómate un momento para relajarte.",
    "Felicidad": "😊 ¡Qué bien! Disfruta este momento y compártelo con alguien especial.",
    "Tristeza": "😢 Está bien sentir tristeza. Habla con alguien de confianza o escribe lo que sientes.",
    "Neutral": "😐 Te ves tranquilo y enfocado. Sigue así, estás en equilibrio."
}

def predecir_cnn(img):
    """Predice la emoción con optimización de memoria - VERSIÓN OPTIMIZADA"""
    try:
        # 🔥 REDUCIR TAMAÑO DE IMAGEN ANTES DE PROCESAR (ahorra memoria)
        if img.shape[0] > 400 or img.shape[1] > 400:
            img = cv2.resize(img, (400, 400))
        
        # Detectar rostro con parámetros más eficientes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 🔥 Reducir resolución para detección más rápida
        gray_small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        
        rostros = face_detector.detectMultiScale(
            gray_small,
            scaleFactor=1.1,  # Más rápido
            minNeighbors=3,   # Menos estricto
            minSize=(30, 30)  # Rostros más pequeños
        )
        
        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            x, y = x*2, y*2  # Escalar de vuelta
            w, h = w*2, h*2
            x, y = max(0, x), max(0, y)
            img = img[y:y+h, x:x+w]
        else:
            print("⚠️ No se detectó rostro, usando imagen completa")
        
        # 🔥 PROCESAR EN RESOLUCIÓN 224x224
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # 🔥 PREDECIR CON BATCH SIZE 1
        pred = modelo_cnn.predict(img, verbose=0, batch_size=1)
        
        indice = np.argmax(pred[0])
        confianza = np.max(pred[0])
        emocion = EMOCIONES[indice] if indice < len(EMOCIONES) else "Desconocida"
        
        print(f"📊 Predicción: {emocion} ({confianza:.2%})")
        
        # 🔥 LIMPIAR MEMORIA
        del img, pred, gray, gray_small, rostros
        import gc
        gc.collect()
        
        return emocion
        
    except Exception as e:
        print(f"❌ Error en predecir_cnn: {e}")
        return "Neutral"

def consejo_emocion(emocion):
    """Obtiene un consejo según la emoción"""
    return CONSEJOS.get(emocion, "🔍 No fue posible determinar la emoción. ¡Cuida de ti mismo!")

# -----------------------------
# RUTAS - COMPLETAMENTE CORREGIDAS
# -----------------------------

# Página principal (login)
@app.route("/")
def index():
    return render_template("login.html")

# Ruta GET para /login (redirige a la raíz)
@app.route("/login", methods=["GET"])
def login_get():
    return redirect("/")

# Ruta POST para procesar el login
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

# Página de registro
@app.route("/register")
def register():
    return render_template("register.html")

# Guardar nuevo usuario
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

# Página principal de la app
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

# Analizar emoción desde cámara
@app.route("/analizar", methods=["POST"])
def analizar():
    if "user" not in session:
        return jsonify({"error": "No hay sesión activa"}), 401
    
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400
        
        # Decodificar imagen base64
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
        
        return jsonify({
            "emotion": emocion,
            "advice": consejo
        })
        
    except Exception as e:
        print("❌ ERROR ANALIZAR:", str(e))
        return jsonify({"error": str(e)}), 500

# Dashboard de estadísticas
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
    
    # Inicializar todas las emociones en 0
    conteo = {emocion: 0 for emocion in EMOCIONES}
    
    for row in datos:
        emocion = row["emocion"]
        cantidad = row[1]
        if emocion in conteo:
            conteo[emocion] = cantidad
    
    return render_template("dashboard.html", conteo=conteo)

# Cerrar sesión
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route('/imagen')
def imagen():
    return render_template('imagen.html')

# Ruta para predecir desde imagen subida
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        # Verificar imagen
        if 'imagen' not in request.files:
            return jsonify({
                'estado': 'error',
                'detalle': 'No se envió imagen'
            }), 400
        
        archivo = request.files['imagen']
        
        # CORREGIDO: Sangría correcta
        if archivo.filename == "":
            return jsonify({
                "estado": "error",
                "detalle": "No se seleccionó ninguna imagen"
            }), 400
        
        # Convertir imagen
        file_bytes = np.frombuffer(archivo.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Validar imagen
        if img is None:
            return jsonify({
                'estado': 'error',
                'detalle': 'Error al leer imagen'
            }), 400
        
        emocion = predecir_cnn(img)
        consejo = consejo_emocion(emocion)
        
        # Guardar en BD si hay sesión
        if "user" in session:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion) VALUES (?, ?)",
                (session["user"], emocion)
            )
            conn.commit()
            conn.close()
        
        return jsonify({
            'emocion': emocion,
            'consejo': consejo,
            'estado': 'success'
        })
        
    except Exception as e:
        print("❌ ERROR predict_image:", str(e))
        return jsonify({
            'emocion': 'No se pudo analizar la imagen',
            'detalle': str(e),
            'estado': 'error'
        }), 500

# Agregar al final de app.py, antes del entrypoint

# -----------------------------
# HEALTH CHECK PARA RENDER
# -----------------------------
@app.route('/health')
def health():
    """Endpoint para health check de Render"""
    return jsonify({"status": "healthy", "message": "EmotiScan funcionando"}), 200

@app.route('/ping')
def ping():
    """Endpoint simple para mantener vivo el servicio"""
    return "pong", 200

# -----------------------------
# MANEJO DE ERRORES GLOBAL
# -----------------------------
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
    app.run(host="0.0.0.0", port=port, debug=False)