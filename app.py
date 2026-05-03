import os

# 🔥 Configuración para evitar errores en Render
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['DEEPFACE_HOME'] = '/tmp'

from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector
from mysql.connector import Error

# -----------------------------
# APP FLASK
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "emocionesIA")

# -----------------------------
# CONEXIÓN MYSQL (MEJORADA)
# -----------------------------
def conectar():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQLHOST", "localhost"),
            user=os.getenv("MYSQLUSER", "root"),
            password=os.getenv("MYSQLPASSWORD", ""),  # 🔥 Cambia en Render
            database=os.getenv("MYSQLDATABASE", "emotiscan"),
            port=int(os.getenv("MYSQLPORT", 3306))
        )
        return conn
    except Error as e:
        print(f"❌ Error conexión MySQL: {e}")
        return None

# -----------------------------
# CREAR BD (MEJORADO)
# -----------------------------
def crear_bd():
    conn = conectar()
    if not conn:
        print("❌ No se pudo conectar a MySQL")
        return
    
    try:
        cursor = conn.cursor()
        
        # Tabla usuarios (con password hasheada)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios(
            id INT AUTO_INCREMENT PRIMARY KEY,
            usuario VARCHAR(100) UNIQUE,
            password VARCHAR(255)  -- Más largo para hash
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emociones(
            id INT AUTO_INCREMENT PRIMARY KEY,
            usuario VARCHAR(100),
            emocion VARCHAR(50),
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("✅ Base de datos MySQL verificada/creada")
    except Exception as e:
        print("❌ Error creando BD:", e)
    finally:
        if conn:
            conn.close()

crear_bd()

# -----------------------------
# DETECTOR DE ROSTRO
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -----------------------------
# CONSEJOS IA
# -----------------------------
def consejo_emocion(emocion):
    consejos = {
        "happy": "😊 ¡Estás feliz! Aprovecha esta energía para avanzar en tus metas.",
        "sad": "😢 Estás triste. Habla con alguien o tómate un descanso.",
        "angry": "😡 Respira profundo. Calma tu mente antes de actuar.",
        "neutral": "😐 Estás estable. Buen momento para concentrarte.",
        "fear": "😨 Tranquilo, todo problema tiene solución.",
        "surprise": "😲 Algo te sorprendió. Analiza con calma la situación."
    }
    return consejos.get(emocion, "Cuida tu bienestar emocional.")

# -----------------------------
# LOGIN (CON HASH)
# -----------------------------
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def validar():
    usuario = request.form.get("usuario")
    password = request.form.get("password")
    
    if not usuario or not password:
        return "❌ Usuario y contraseña son requeridos", 400
    
    conn = conectar()
    if not conn:
        return "❌ Error de conexión a la base de datos", 500
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password FROM usuarios WHERE usuario=%s",
            (usuario,)
        )
        result = cursor.fetchone()
        
        if result and check_password_hash(result[0], password):
            session["user"] = usuario
            return redirect("/inicio")
        else:
            return "❌ Usuario o contraseña incorrectos", 401
    except Exception as e:
        print(f"Error en login: {e}")
        return "❌ Error interno", 500
    finally:
        conn.close()

# -----------------------------
# REGISTRO (CON HASH)
# -----------------------------
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/guardar", methods=["POST"])
def guardar_usuario():
    usuario = request.form.get("usuario")
    password = request.form.get("password")
    
    if not usuario or not password:
        return "❌ Usuario y contraseña son requeridos", 400
    
    conn = conectar()
    if not conn:
        return "❌ Error de conexión a la base de datos", 500
    
    try:
        cursor = conn.cursor()
        # Hashear contraseña
        hashed_password = generate_password_hash(password)
        
        cursor.execute(
            "INSERT INTO usuarios (usuario, password) VALUES (%s, %s)",
            (usuario, hashed_password)
        )
        conn.commit()
        return redirect("/")
    except mysql.connector.IntegrityError:
        return "⚠️ El usuario ya existe. <a href='/register'>Intentar de nuevo</a>", 409
    except Exception as e:
        print(f"Error en registro: {e}")
        return "❌ Error interno del servidor", 500
    finally:
        conn.close()

# -----------------------------
# INICIO
# -----------------------------
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

# -----------------------------
# ANALIZAR EMOCIÓN
# -----------------------------
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
        
        # DeepFace
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        emocion = result[0]["dominant_emotion"]
        consejo = consejo_emocion(emocion)
        
        # Guardar en BD
        conn = conectar()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion) VALUES (%s, %s)",
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

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    
    conn = conectar()
    if not conn:
        return "❌ Error de conexión", 500
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT emocion, COUNT(*)
            FROM emociones
            WHERE usuario=%s
            GROUP BY emocion
        """, (session["user"],))
        
        datos = cursor.fetchall()
        
        conteo = {
            "happy": 0, "sad": 0, "angry": 0,
            "neutral": 0, "fear": 0, "surprise": 0
        }
        
        for emocion, cantidad in datos:
            conteo[emocion] = cantidad
        
        return render_template("dashboard.html", conteo=conteo)
    except Exception as e:
        print(f"Error dashboard: {e}")
        return "❌ Error interno", 500
    finally:
        conn.close()

# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# -----------------------------
# ENTRYPOINT
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)