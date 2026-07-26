from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import cv2
import numpy as np
import base64
import sqlite3
import logging
import gc
import sys
import time

# 🔥 CONFIGURACIÓN DE MEMORIA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos livianos
from face_detector import detectar_rostro
from predict import predecir
from mensajes import obtener_mensaje, mostrar_estado_mensajes

# APP FLASK
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "emocionesIA_segura_2026")

# BASE DE DATOS
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
            mensaje TEXT,
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("✅ Base de datos inicializada")

init_db()

# CONSEJOS
CONSEJOS = {
    "Enojo": "😡 Respira profundamente y cuenta hasta 10.",
    "Felicidad": "😊 ¡Qué bien! Disfruta este momento.",
    "Tristeza": "😢 Habla con alguien de confianza.",
    "Sorpresa": "😮 Tómate un momento para procesarlo.",
    "Neutral": "😐 Estás en equilibrio."
}

def predecir_cnn(img):
    """Función principal con timeout y manejo de memoria"""
    try:
        if img is None:
            return "Neutral"
        
        # Limitar tiempo de procesamiento
        start_time = time.time()
        
        # Detectar rostro
        rostro = detectar_rostro(img)
        if rostro is None:
            return "Neutral"
        
        # Si pasó mucho tiempo, retornar Neutral
        if time.time() - start_time > 2.0:
            logger.warning("⏰ Timeout en procesamiento")
            return "Neutral"
        
        # Predecir
        emocion, _, _ = predecir(rostro)
        
        # Liberar memoria
        gc.collect()
        
        return emocion
        
    except Exception as e:
        logger.error(f"❌ Error en predecir_cnn: {e}")
        gc.collect()
        return "Neutral"

# =============================
# RUTAS DE FLASK
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
        logger.error(f"Error en registro: {e}")
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
    """Endpoint principal con manejo de errores robusto"""
    if "user" not in session:
        return jsonify({"error": "No hay sesión activa"}), 401

    try:
        # Verificar contenido
        data = request.get_json()
        if data is None or "image" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        # Decodificar imagen
        try:
            image_data = data["image"].split(";base64,")[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decodificando imagen: {e}")
            return jsonify({"error": "Error al procesar la imagen"}), 400

        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400

        # Analizar con timeout
        emocion = predecir_cnn(img)
        mensaje_aleatorio = obtener_mensaje(emocion)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")

        # Guardar en BD
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion, mensaje) VALUES (?, ?, ?)",
                (session["user"], emocion, mensaje_aleatorio)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error guardando en BD: {e}")

        # Liberar memoria
        gc.collect()

        return jsonify({
            "success": True,
            "emotion": emocion,
            "advice": consejo,
            "message": mensaje_aleatorio
        })

    except Exception as e:
        logger.error(f"❌ Error en /analizar: {e}")
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
    
    EMOCIONES = ["Enojo", "Felicidad", "Tristeza", "Sorpresa", "Neutral"]
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
        mensaje_aleatorio = obtener_mensaje(emocion)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")
        
        if "user" in session:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion, mensaje) VALUES (?, ?, ?)",
                (session["user"], emocion, mensaje_aleatorio)
            )
            conn.commit()
            conn.close()
        
        gc.collect()
        
        return jsonify({
            'emocion': emocion,
            'consejo': consejo,
            'mensaje': mensaje_aleatorio,
            'estado': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Error predict_image: {e}")
        gc.collect()
        return jsonify({'estado': 'error', 'detalle': str(e)}), 500

@app.route('/health')
def health():
    """Health check para Render"""
    return jsonify({
        "status": "healthy",
        "memory": "optimized",
        "version": "2.0-lite"
    }), 200

@app.route('/debug')
def debug():
    """Endpoint para debug"""
    import sys
    return jsonify({
        "python": sys.version,
        "memory_optimized": True,
        "tensorflow_loaded": False
    })

# =============================
# ENTRYPOINT
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Servidor iniciado en puerto {port}")
    logger.info("📦 Versión: Optimizada para memoria (sin TensorFlow)")
    app.run(host="0.0.0.0", port=port, debug=False)