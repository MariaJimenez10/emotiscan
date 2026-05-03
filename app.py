import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime
from flask import Flask, render_template, request, redirect, session, jsonify
import base64
import cv2
import numpy as np
from deepface import DeepFace
import mysql.connector

# -----------------------------
# APP FLASK
# -----------------------------
app = Flask(__name__)
app.secret_key = "emocionesIA"

# -----------------------------
# CONEXIÓN MYSQL
# -----------------------------
def conectar():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "localhost"),
        user=os.getenv("MYSQLUSER", "root"),
        password=os.getenv("MYSQLPASSWORD", "123"),
        database=os.getenv("MYSQLDATABASE", "emotiscan"),
        port=int(os.getenv("MYSQLPORT", 3306))
    )

# -----------------------------
# CREAR BD
# -----------------------------
def crear_bd():
    try:
        conn = conectar()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios(
            id INT AUTO_INCREMENT PRIMARY KEY,
            usuario VARCHAR(100) UNIQUE,
            password VARCHAR(100)
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
    finally:
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
        "happy": "Estás feliz 😊 aprovecha para avanzar en tus metas.",
        "sad": "Estás triste 😢 habla con alguien o descansa un poco.",
        "angry": "Respira profundo 😡 calma tu mente antes de actuar.",
        "neutral": "Estás estable 😐 buen momento para concentrarte.",
        "fear": "Tranquilo 😨 todo problema tiene solución.",
        "surprise": "Algo te sorprendió 😲 analiza con calma la situación."
    }
    return consejos.get(emocion, "Cuida tu bienestar emocional.")

# -----------------------------
# LOGIN
# -----------------------------
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def validar():
    usuario = request.form["usuario"]
    password = request.form["password"]

    conn = conectar()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM usuarios WHERE usuario=%s AND password=%s",
        (usuario, password)
    )
    user = cursor.fetchone()
    conn.close()

    if user:
        session["user"] = usuario
        return redirect("/inicio")

    return "❌ Usuario o contraseña incorrectos"

# -----------------------------
# REGISTRO
# -----------------------------
@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/guardar", methods=["POST"])
def guardar_usuario():
    usuario = request.form["usuario"]
    password = request.form["password"]

    conn = conectar()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO usuarios (usuario,password) VALUES (%s,%s)",
            (usuario, password)
        )
        conn.commit()
        return redirect("/")
    except:
        return "⚠️ El usuario ya existe"
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
        image_data = data["image"].split(";base64,")[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detección de rostro (opcional)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Análisis con IA
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
        return jsonify({"error": str(e)}), 500

# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT emocion, COUNT(*)
        FROM emociones
        WHERE usuario=%s
        GROUP BY emocion
    """, (session["user"],))

    datos = cursor.fetchall()
    conn.close()

    conteo = {
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "neutral": 0,
        "fear": 0,
        "surprise": 0
    }

    for emocion, cantidad in datos:
        conteo[emocion] = cantidad

    return render_template("dashboard.html", conteo=conteo)

# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)