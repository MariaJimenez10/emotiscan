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
