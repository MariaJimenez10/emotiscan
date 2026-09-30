from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from flask import send_from_directory
import os
import cv2
import base64
import logging
import numpy as np
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect
from flask_cors import CORS


# ==========================================================
# IMPORTAR MÓDULOS LOCALES
# ==========================================================

import face_detector
import predict


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

app = Flask(__name__)

app.secret_key = "tu_clave_secreta_aqui"

CORS(app)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "emotiscan.db")

DEBUG_DIR = os.path.join(BASE_DIR, "debug_rostros")

os.makedirs(DEBUG_DIR, exist_ok=True)


# ==========================================================
# BASE DE DATOS
# ==========================================================

def init_db():

    try:

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()

        # --------------------------------------------------
        # USUARIOS
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --------------------------------------------------
        # EMOCIONES
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emociones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER NOT NULL,
                emocion TEXT NOT NULL,
                confianza REAL NOT NULL,
                todas_emociones TEXT,
                fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
            )
        """)

        conn.commit()

        conn.close()

        logger.info("✅ Base de datos inicializada")

    except Exception as e:

        logger.error(
            f"❌ Error inicializando BD: {e}"
        )


def get_db_connection():

    return sqlite3.connect(DB_PATH)


# ==========================================================
# PÁGINA DE INICIO
# ==========================================================

@app.route("/")
def index():

    # Si ya inició sesión, mostramos el inicio
    # con acceso a la detección.

    if "usuario_id" in session:

        return render_template(
            "inicio.html",
            nombre=session.get(
                "usuario_nombre",
                "Usuario"
            )
        )

    # Si no ha iniciado sesión,
    # también mostramos la portada.

    return render_template(
        "inicio.html",
        nombre=None
    )


# ==========================================================
# REGISTRO - PÁGINA
# ==========================================================

@app.route("/registro", methods=["GET"])
def registro_pagina():

    return render_template(
        "registro.html"
    )


# ==========================================================
# REGISTRO - PROCESAR
# ==========================================================

@app.route("/registro", methods=["POST"])
def registro():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error": "No se recibieron datos"

            }), 400


        nombre = data.get("nombre")

        email = data.get("email")

        password = data.get("password")


        if not nombre or not email or not password:

            return jsonify({

                "success": False,

                "error": "Todos los campos son obligatorios"

            }), 400


        conn = get_db_connection()

        cursor = conn.cursor()


        cursor.execute(
            """
            INSERT INTO usuarios
            (nombre, email, password)
            VALUES (?, ?, ?)
            """,
            (
                nombre,
                email,
                password
            )
        )


        conn.commit()

        conn.close()


        logger.info(
            f"✅ Usuario registrado: {email}"
        )


        return jsonify({

            "success": True,

            "mensaje":
                "Registro exitoso. Ahora inicia sesión."

        })


    except sqlite3.IntegrityError:

        return jsonify({

            "success": False,

            "error":
                "El email ya está registrado"

        }), 400


    except Exception as e:

        logger.exception(
            "❌ Error en registro"
        )


        return jsonify({

            "success": False,

            "error":
                "Error interno del servidor"

        }), 500


# ==========================================================
# LOGIN - PÁGINA
# ==========================================================

@app.route("/login", methods=["GET"])
def login_pagina():

    return render_template(
        "login.html"
    )


# ==========================================================
# LOGIN - PROCESAR
# ==========================================================

@app.route("/login", methods=["POST"])
def login():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No se recibieron datos"

            }), 400


        email = data.get("email")

        password = data.get("password")


        if not email or not password:

            return jsonify({

                "success": False,

                "error":
                    "Email y contraseña son obligatorios"

            }), 400


        conn = get_db_connection()

        cursor = conn.cursor()


        cursor.execute(
            """
            SELECT id, nombre
            FROM usuarios
            WHERE email = ?
            AND password = ?
            """,
            (
                email,
                password
            )
        )


        usuario = cursor.fetchone()

        conn.close()


        if usuario:

            session["usuario_id"] = usuario[0]

            session["usuario_nombre"] = usuario[1]


            logger.info(
                f"✅ Login correcto: {email}"
            )


            return jsonify({

                "success": True,

                "nombre": usuario[1],

                "redirect": "/inicio"

            })


        return jsonify({

            "success": False,

            "error":
                "Credenciales incorrectas"

        })


    except Exception as e:

        logger.exception(
            "❌ Error en login"
        )


        return jsonify({

            "success": False,

            "error":
                "Error interno del servidor"

        }), 500


# ==========================================================
# INICIO DESPUÉS DEL LOGIN
# ==========================================================

@app.route("/inicio")
def inicio():

    # ------------------------------------------------------
    # COMPROBAR SESIÓN
    # ------------------------------------------------------

    if "usuario_id" not in session:

        return redirect("/login")


    return render_template(

        "inicio.html",

        nombre=session.get(
            "usuario_nombre",
            "Usuario"
        )

    )


# ==========================================================
# CÁMARA / DETECCIÓN
# ==========================================================

@app.route("/camara")
def camara():

    # ------------------------------------------------------
    # PROTEGER LA CÁMARA
    # ------------------------------------------------------

    if "usuario_id" not in session:

        logger.warning(
            "⚠️ Intento de acceder a cámara sin login"
        )

        return redirect("/login")


    # ------------------------------------------------------
    # USUARIO AUTENTICADO
    # ------------------------------------------------------

    return render_template(
        "index.html"
    )
# ==========================================================
# HISTORIAL DE EMOCIONES
# ==========================================================

@app.route("/historial")
def historial():

    # Comprobar que el usuario haya iniciado sesión
    if "usuario_id" not in session:
        return redirect("/login")

    try:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row

        registros = conn.execute(
            """
            SELECT id, emocion, confianza, todas_emociones, fecha
            FROM emociones
            WHERE usuario_id = ?
            ORDER BY fecha DESC
            """,
            (session["usuario_id"],)
        ).fetchall()

        conn.close()

        return render_template(
            "historial.html",
            registros=registros,
            nombre=session.get("usuario_nombre", "Usuario")
        )

    except Exception as e:
        logger.exception(
            f"❌ Error cargando historial: {e}"
        )

        return "Error al cargar el historial", 500


# ==========================================================
# LOGOUT
# ==========================================================
import base64
import sqlite3
import logging
import gc
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos
from face_detector import detectar_rostro
from predict import predecir
from mensajes import obtener_mensaje

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

    try:

        logger.info("======================================")
        logger.info("🧠 INICIANDO ANÁLISIS CNN")
        logger.info(f"Imagen recibida: {img.shape}")
        logger.info(f"Tipo: {img.dtype}")

        # ==========================================
        # DETECTAR ROSTRO
        # ==========================================

        rostro = detectar_rostro(img)

        if rostro is None:

            logger.warning(
                "❌ No se pudo detectar rostro"
            )

            return "Neutral"


        logger.info(
            f"✅ Rostro enviado a predict(): "
            f"{rostro.shape}"
        )


        # ==========================================
        # PREDICCIÓN
        # ==========================================

        emocion, confianza, predicciones = predecir(
            rostro
        )


        logger.info(
            f"🎯 Resultado: {emocion}"
        )

        logger.info(
            f"🎯 Confianza: {confianza:.4f}"
        )

        logger.info(
            f"🎯 Predicciones: {predicciones}"
        )


        logger.info("======================================")


        return emocion


    except Exception as e:

        logger.exception(
            f"❌ Error en predecir_cnn: {e}"
        )

        return "Neutral"

# =============================
# RUTAS
# =============================

@app.route("/")
def index():
    error = request.args.get('error')
    return render_template("login.html", error=error)

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
        return redirect("/?error=invalid")

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
        logger.error(f"Error: {e}")
        return "❌ Error interno", 500

@app.route("/inicio")
def inicio():
    if "user" not in session:
        return redirect("/")
    
    ahora = datetime.now()
    return render_template(
        "index.html",  # ← ESTE ES EL HTML CON LA MALLA FACIAL
        usuario=session["user"],
        fecha=ahora.strftime("%Y-%m-%d"),
        hora=ahora.strftime("%H:%M:%S")
    )

@app.route("/analizar", methods=["POST"])
def analizar():
    logger.info("📥 /analizar")
    
    if "user" not in session:
        return jsonify({"error": "No hay sesión"}), 401

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        # Decodificar imagen
        try:
            image_data = data["image"].split(";base64,")[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"❌ Error decodificando: {e}")
            return jsonify({"error": "Error procesando imagen"}), 400

        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400

        emocion = predecir_cnn(img)
        mensaje = obtener_mensaje(emocion)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")

        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion, mensaje) VALUES (?, ?, ?)",
                (session["user"], emocion, mensaje)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Error BD: {e}")

        gc.collect()
        logger.info(f"✅ {emocion}")
        
        return jsonify({
            "success": True,
            "emotion": emocion,
            "advice": consejo,
            "message": mensaje
        })

    except Exception as e:
        logger.error(f"❌ Error: {e}")
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

    nombre = session.get(
        "usuario_nombre",
        "Usuario"
    )

    session.clear()


    logger.info(
        f"👋 Sesión cerrada: {nombre}"
    )


    return redirect("/")

# ==========================================================
# ANALIZAR EMOCIÓN
# ==========================================================

@app.route("/analizar", methods=["POST"])
def analizar():

    try:

        logger.info("=" * 60)

        logger.info("📥 /analizar")

        logger.info(
            "🧠 INICIANDO ANÁLISIS"
        )


        # ==================================================
        # 1. COMPROBAR SESIÓN
        # ==================================================

        if "usuario_id" not in session:

            logger.warning(
                "⚠️ Análisis rechazado: usuario no autenticado"
            )

            return jsonify({

                "success": False,

                "error":
                    "Debes iniciar sesión para analizar emociones."

            }), 401


        # ==================================================
        # 2. OBTENER JSON
        # ==================================================

        if not request.is_json:

            logger.error(
                "❌ La petición no es JSON"
            )

            return jsonify({

                "success": False,

                "error":
                    "La petición debe ser JSON"

            }), 400


        data = request.get_json(
            silent=True
        )


        if not data:

            logger.error(
                "❌ JSON vacío"
            )

            return jsonify({

                "success": False,

                "error":
                    "JSON vacío"

            }), 400


        logger.info(
            f"📦 Campos JSON: {list(data.keys())}"
        )


        # ==================================================
        # 3. OBTENER IMAGEN
        # ==================================================

        imagen_base64 = data.get(
            "image"
        )


        if imagen_base64 is None:

            imagen_base64 = data.get(
                "imagen"
            )


        if not imagen_base64:

            logger.error(
                "❌ No se recibió imagen"
            )

            return jsonify({

                "success": False,

                "error":
                    "No se recibió imagen"

            }), 400


        logger.info(
            "✅ Campo de imagen encontrado"
        )


        # ==================================================
        # 4. DECODIFICAR IMAGEN
        # ==================================================

        try:

            if "," in imagen_base64:

                imagen_base64 = \
                    imagen_base64.split(
                        ",",
                        1
                    )[1]


            img_bytes = base64.b64decode(

                imagen_base64,

                validate=True

            )


            nparr = np.frombuffer(

                img_bytes,

                dtype=np.uint8

            )


            frame = cv2.imdecode(

                nparr,

                cv2.IMREAD_COLOR

            )


        except Exception as e:

            logger.error(

                f"❌ Error decodificando imagen: {e}"

            )

            return jsonify({

                "success": False,

                "error":
                    "Imagen inválida"

            }), 400


        if frame is None:

            logger.error(

                "❌ cv2.imdecode devolvió None"

            )

            return jsonify({

                "success": False,

                "error":
                    "No se pudo decodificar la imagen"

            }), 400


        logger.info(

            f"📷 Imagen recibida: {frame.shape}"

        )


        logger.info(

            f"💾 Tipo: {frame.dtype}"

        )


        # ==================================================
        # 5. GUARDAR IMAGEN ORIGINAL
        # ==================================================

        original_path = os.path.join(

            DEBUG_DIR,

            "frame_recibido.jpg"

        )


        cv2.imwrite(

            original_path,

            frame

        )


        logger.info(

            f"💾 Imagen guardada: {original_path}"

        )


        # ==================================================
        # 6. DETECTAR ROSTRO
        # ==================================================

        logger.info(

            "👤 Detectando rostro..."

        )


        rostro = face_detector.detectar_rostro(

            frame

        )


        if rostro is None:

            logger.warning(

                "⚠️ No se detectó ningún rostro"

            )


            return jsonify({

                "success": False,

                "error":
                    "No se detectó ningún rostro",

                "emocion":
                    "No detectado",

                "confianza":
                    0,

                "todas":
                    {}

            })


        logger.info(

            f"✅ ROSTRO DETECTADO: {rostro.shape}"

        )


        # ==================================================
        # 7. GUARDAR ROSTRO
        # ==================================================

        rostro_path = os.path.join(

            DEBUG_DIR,

            "rostro_analizar.jpg"

        )


        cv2.imwrite(

            rostro_path,

            rostro

        )


        logger.info(

            f"💾 Rostro guardado: {rostro_path}"

        )


        # ==================================================
        # 8. PREDECIR EMOCIÓN
        # ==================================================

        logger.info(

            "🧠 PREDICIENDO EMOCIÓN..."

        )


        emocion, confianza, todas, consejo = \
            predict.predecir(

                rostro

            )


        logger.info(

            f"🎯 EMOCIÓN: {emocion}"

        )


        logger.info(

            f"📊 CONFIANZA: {confianza:.2f}%"

        )


        logger.info(

            f"📊 TODAS: {todas}"

        )


        # ==================================================
        # 9. GUARDAR EN BASE DE DATOS
        # ==================================================

        try:

            conn = get_db_connection()

            cursor = conn.cursor()


            cursor.execute(
                """
                INSERT INTO emociones
                (
                    usuario_id,
                    emocion,
                    confianza,
                    todas_emociones
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    session["usuario_id"],
                    emocion,
                    confianza,
                    str(todas)
                )
            )


            conn.commit()

            conn.close()


            logger.info(
                "✅ Análisis guardado en BD"
            )


        except Exception as e:

            logger.error(

                f"❌ Error guardando BD: {e}"

            )


        # ==================================================
        # 10. RESPUESTA
        # ==================================================

        logger.info(
            "✅ ANÁLISIS COMPLETADO"
        )

        logger.info("=" * 60)


        return jsonify({

            "success": True,

            "emocion": emocion,

            "confianza": confianza,

            "todas": todas,

            "consejo": consejo

        })


    except Exception as e:

        logger.exception(
            "❌ ERROR GENERAL EN /analizar"
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================================
# INICIAR SERVIDOR
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # BASE DE DATOS
    # ------------------------------------------------------

    init_db()


    # ------------------------------------------------------
    # CARGAR MODELO
    # ------------------------------------------------------

    try:

        predict.cargar_modelo()

        logger.info(
            "✅ Modelo cargado correctamente"
        )

    except Exception as e:

        logger.exception(
            "❌ Error cargando modelo"
        )


    # ------------------------------------------------------
    # SERVIDOR
    # ------------------------------------------------------

    logger.info(
        "🚀 Servidor en puerto 10000"
    )


    app.run(

        host="0.0.0.0",

        port=10000,

        debug=False

    )
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
        mensaje = obtener_mensaje(emocion)
        consejo = CONSEJOS.get(emocion, "Cuida de ti mismo.")
        
        if "user" in session:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO emociones (usuario, emocion, mensaje) VALUES (?, ?, ?)",
                (session["user"], emocion, mensaje)
            )
            conn.commit()
            conn.close()
        
        gc.collect()
        
        return jsonify({
            'emocion': emocion,
            'consejo': consejo,
            'mensaje': mensaje,
            'estado': 'success'
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        gc.collect()
        return jsonify({'estado': 'error', 'detalle': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "version": "ultra-lite"
    }), 200

# ============================================
# NUEVA RUTA PARA STATIC FILES (OPCIONAL)
# ============================================
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Servidor en puerto {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
