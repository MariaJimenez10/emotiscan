import os
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify  # ✅ redirect ya está aquí
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
import sqlite3

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
# CARGAR MODELO RESTNEST
# -----------------------------

modelo_cnn = load_model("modelo_resnet50_emociones.h5")
print(modelo_cnn.summary())

emociones_cnn = [
    "Enojo",
    "Felicidad",
    "Neutral",
    "Tristeza"
]

print("✅ Modelo ResNet50 cargado correctamente")

# =============================
# FUNCION CNN
# =============================

from tensorflow.keras.applications.resnet50 import preprocess_input

def predecir_cnn(img):

    imagen = cv2.resize(img, (224, 224))

    imagen = np.expand_dims(imagen.astype(np.float32), axis=0)

    imagen = preprocess_input(imagen)

    prediccion = modelo_cnn.predict(imagen, verbose=0)

    clase = np.argmax(prediccion)

    emociones = [
        "Enojo",
        "Felicidad",
        "Neutral",
        "Tristeza"
    ]

    return emociones[clase]

# -----------------------------
# CONSEJOS IA
# -----------------------------
def consejo_emocion(emocion):

    consejos = {

        "Felicidad":
        "😊 Te ves feliz. Continúa disfrutando este momento.",

        "Tristeza":
        "😢 Parece que te sientes triste. Puedes hablar con una persona de confianza.",

        "Enojo":
        "😡 Respira lentamente y toma un momento para relajarte.",

        "Neutral":
        "😐 Te ves tranquilo y concentrado."
    }

    return consejos.get(
        emocion,
        "🔍 No fue posible determinar la emoción."
    )
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

    if result and check_password_hash(result[0],password):
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

# Analizar emoción
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
    
    conteo = {
    "Enojo": 0,
    "Felicidad": 0,
    "Neutral": 0,
    "Tristeza": 0
}
    
    for row in datos:
        emocion = row["emocion"]
        cantidad = row[1]
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

####
@app.route('/predict_image', methods=['POST'])
def predict_image():

    try:

        # Verificar imagen
        if 'imagen' not in request.files:
            return jsonify({
                'emocion': 'No se envió imagen'
            }), 400

        archivo = request.files['imagen']

        # Convertir imagen
        file_bytes = np.frombuffer(
            archivo.read(),
            np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        # Validar imagen
        if img is None:
            return jsonify({
                'emocion': 'Error al leer imagen'
            }), 400

        emocion = predecir_cnn(img)
        consejo = consejo_emocion(emocion)

        # Guardar en BD
        if "user" in session:

            conn = get_db()

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO emociones
                (usuario, emocion)
                VALUES (?, ?)
                """,
                (
                    session["user"],
                    emocion
                )
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
# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)