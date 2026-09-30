from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    jsonify,
    send_from_directory
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from datetime import datetime

from flask_cors import CORS

import os
import cv2
import base64
import logging
import numpy as np
import sqlite3
import json
import gc


# ==========================================================
# CONFIGURACIÓN DE LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ==========================================================
# IMPORTAR MÓDULOS DEL PROYECTO
# ==========================================================

import face_detector
import predict

from mensajes import obtener_mensaje


# ==========================================================
# CONFIGURACIÓN FLASK
# ==========================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "SECRET_KEY",
    "emocionesIA_segura_2026"
)

CORS(app)


# ==========================================================
# DIRECTORIO PRINCIPAL
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==========================================================
# BASE DE DATOS
# ==========================================================

if os.environ.get("RENDER"):

    DB_PATH = "/tmp/emotiscan.db"

else:

    DB_PATH = os.path.join(
        BASE_DIR,
        "emotiscan.db"
    )


# ==========================================================
# DIRECTORIOS
# ==========================================================

DEBUG_DIR = os.path.join(
    BASE_DIR,
    "debug_rostros"
)

ROSTROS_DIR = os.path.join(
    BASE_DIR,
    "rostros_detectados"
)


os.makedirs(
    DEBUG_DIR,
    exist_ok=True
)

os.makedirs(
    ROSTROS_DIR,
    exist_ok=True
)


# ==========================================================
# CONSEJOS
# ==========================================================

CONSEJOS = {

    "Enojo":
        "😡 Respira profundamente y cuenta hasta 10.",

    "Felicidad":
        "😊 ¡Qué bien! Disfruta este momento.",

    "Tristeza":
        "😢 Habla con alguien de confianza.",

    "Sorpresa":
        "😮 Tómate un momento para procesarlo.",

    "Neutral":
        "😐 Estás en equilibrio.",

    "Furia":
        "😡 Respira profundamente y cuenta hasta 10.",

    "Alegria":
        "😊 ¡Qué bien! Disfruta este momento."

}


# ==========================================================
# CONEXIÓN BASE DE DATOS
# ==========================================================

def get_db_connection():

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    return conn


# ==========================================================
# INICIALIZAR BASE DE DATOS
# ==========================================================

def init_db():

    try:

        conn = get_db_connection()

        cursor = conn.cursor()

        # ==================================================
        # TABLA USUARIOS
        # ==================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                nombre TEXT,

                email TEXT UNIQUE,

                password TEXT NOT NULL,

                usuario TEXT UNIQUE,

                fecha_creacion
                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )
        """)

        # ==================================================
        # TABLA EMOCIONES
        # ==================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emociones (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                usuario_id INTEGER,

                usuario TEXT,

                emocion TEXT NOT NULL,

                confianza REAL DEFAULT 0,

                todas_emociones TEXT,

                mensaje TEXT,

                fecha
                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                fecha_registro
                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(usuario_id)
                    REFERENCES usuarios(id)

            )
        """)

        conn.commit()

        conn.close()

        logger.info(
            "✅ Base de datos inicializada correctamente"
        )

    except Exception as e:

        logger.exception(
            f"❌ Error inicializando base de datos: {e}"
        )


# ==========================================================
# COMPROBAR USUARIO AUTENTICADO
# ==========================================================

def usuario_autenticado():

    return (
        session.get("usuario_id")
        or session.get("user")
    )


# ==========================================================
# PÁGINA PRINCIPAL
# ==========================================================

@app.route("/")
def index():

    if usuario_autenticado():

        return redirect(
            "/inicio"
        )

    error = request.args.get(
        "error"
    )

    return render_template(
        "login.html",
        error=error
    )


# ==========================================================
# PÁGINA REGISTRO
# ==========================================================

@app.route(
    "/registro",
    methods=["GET"]
)
def registro_pagina():

    return render_template(
        "registro.html"
    )


# ==========================================================
# RUTA REGISTER
# ==========================================================

@app.route(
    "/register",
    methods=["GET"]
)
def register():

    return redirect(
        "/registro"
    )


# ==========================================================
# PROCESAR REGISTRO
# ==========================================================

@app.route(
    "/registro",
    methods=["POST"]
)
def registro():

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

        nombre = (
            data.get("nombre")
            or data.get("usuario")
        )

        email = data.get(
            "email"
        )

        password = data.get(
            "password"
        )

        # ==================================================
        # VALIDAR NOMBRE
        # ==================================================

        if not nombre:

            return jsonify({

                "success": False,

                "error":
                    "El nombre es obligatorio"

            }), 400

        # ==================================================
        # VALIDAR CONTRASEÑA
        # ==================================================

        if not password:

            return jsonify({

                "success": False,

                "error":
                    "La contraseña es obligatoria"

            }), 400

        # ==================================================
        # GENERAR HASH
        # ==================================================

        password_hash = generate_password_hash(
            password
        )

        # ==================================================
        # GUARDAR USUARIO
        # ==================================================

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO usuarios
            (
                nombre,
                email,
                password,
                usuario
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                nombre,
                email if email else None,
                password_hash,
                nombre
            )
        )

        conn.commit()

        conn.close()

        logger.info(
            f"✅ Usuario registrado: {nombre}"
        )

        return jsonify({

            "success": True,

            "mensaje":
                "Registro exitoso. Ahora inicia sesión.",

            "redirect":
                "/login"

        })

    except sqlite3.IntegrityError:

        return jsonify({

            "success": False,

            "error":
                "El usuario o email ya está registrado"

        }), 400

    except Exception as e:

        logger.exception(
            f"❌ Error en registro: {e}"
        )

        return jsonify({

            "success": False,

            "error":
                "Error interno del servidor"

        }), 500


# ==========================================================
# REGISTRO FORMULARIO ANTIGUO
# ==========================================================

@app.route(
    "/guardar",
    methods=["POST"]
)
def guardar_usuario():

    try:

        usuario = request.form.get(
            "usuario"
        )

        password = request.form.get(
            "password"
        )

        if not usuario or not password:

            return (
                "❌ Usuario y contraseña son requeridos",
                400
            )

        password_hash = generate_password_hash(
            password
        )

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO usuarios
            (
                usuario,
                nombre,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                usuario,
                usuario,
                password_hash
            )
        )

        conn.commit()

        conn.close()

        return redirect(
            "/"
        )

    except sqlite3.IntegrityError:

        return (
            "⚠️ El usuario ya existe. "
            "<a href='/registro'>Intentar de nuevo</a>"
        )

    except Exception as e:

        logger.exception(
            f"❌ Error registrando usuario: {e}"
        )

        return (
            "❌ Error interno",
            500
        )


# ==========================================================
# LOGIN - PÁGINA
# ==========================================================

@app.route(
    "/login",
    methods=["GET"]
)
def login_pagina():

    return render_template(
        "login.html"
    )


# ==========================================================
# LOGIN - PROCESAR
# ==========================================================

@app.route(
    "/login",
    methods=["POST"]
)
def login():

    try:

        # ==================================================
        # LOGIN MEDIANTE JSON
        # ==================================================

        if request.is_json:

            data = request.get_json(
                silent=True
            )

            if not data:

                return jsonify({

                    "success": False,

                    "error":
                        "No se recibieron datos"

                }), 400

            email = data.get(
                "email"
            )

            usuario = data.get(
                "usuario"
            )

            password = data.get(
                "password"
            )

            # ==================================================
            # VALIDAR CONTRASEÑA
            # ==================================================

            if not password:

                return jsonify({

                    "success": False,

                    "error":
                        "La contraseña es obligatoria"

                }), 400

            # ==================================================
            # CONECTAR BD
            # ==================================================

            conn = get_db_connection()

            cursor = conn.cursor()

            # ==================================================
            # BUSCAR POR EMAIL
            # ==================================================

            if email:

                cursor.execute(
                    """
                    SELECT
                        id,
                        nombre,
                        usuario,
                        email,
                        password
                    FROM usuarios
                    WHERE email = ?
                    """,
                    (email,)
                )

            # ==================================================
            # BUSCAR POR USUARIO
            # ==================================================

            elif usuario:

                cursor.execute(
                    """
                    SELECT
                        id,
                        nombre,
                        usuario,
                        email,
                        password
                    FROM usuarios
                    WHERE usuario = ?
                    """,
                    (usuario,)
                )

            else:

                conn.close()

                return jsonify({

                    "success": False,

                    "error":
                        "Debes ingresar usuario o email"

                }), 400

            resultado = cursor.fetchone()

            conn.close()

            # ==================================================
            # COMPROBAR SI EXISTE
            # ==================================================

            if resultado:

                password_guardada = (
                    resultado["password"]
                )

                password_correcta = False

                # ==================================================
                # COMPROBAR PASSWORD HASH
                # ==================================================

                try:

                    password_correcta = check_password_hash(
                        password_guardada,
                        password
                    )

                except Exception:

                    password_correcta = False

                # ==================================================
                # COMPATIBILIDAD PASSWORD ANTIGUA
                # ==================================================

                if not password_correcta:

                    if password_guardada == password:

                        password_correcta = True

                # ==================================================
                # LOGIN CORRECTO
                # ==================================================

                if password_correcta:

                    session.clear()

                    session["usuario_id"] = (
                        resultado["id"]
                    )

                    session["usuario_nombre"] = (
                        resultado["nombre"]
                        or resultado["usuario"]
                        or "Usuario"
                    )

                    session["user"] = (
                        resultado["usuario"]
                        or resultado["nombre"]
                        or "Usuario"
                    )

                    logger.info(
                        "✅ Login correcto"
                    )

                    return jsonify({

                        "success": True,

                        "nombre":
                            session["usuario_nombre"],

                        "redirect":
                            "/inicio"

                    })

            # ==================================================
            # CREDENCIALES INCORRECTAS
            # ==================================================

            return jsonify({

                "success": False,

                "error":
                    "Credenciales incorrectas"

            }), 401

        # ==================================================
        # LOGIN MEDIANTE FORMULARIO
        # ==================================================

        usuario = request.form.get(
            "usuario"
        )

        password = request.form.get(
            "password"
        )

        if not usuario or not password:

            return (
                "❌ Usuario y contraseña son requeridos",
                400
            )

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                nombre,
                usuario,
                password
            FROM usuarios
            WHERE usuario = ?
            """,
            (usuario,)
        )

        resultado = cursor.fetchone()

        conn.close()

        # ==================================================
        # COMPROBAR USUARIO
        # ==================================================

        if resultado:

            password_guardada = (
                resultado["password"]
            )

            password_correcta = False

            # ==================================================
            # COMPROBAR HASH
            # ==================================================

            try:

                password_correcta = check_password_hash(
                    password_guardada,
                    password
                )

            except Exception:

                password_correcta = False

            # ==================================================
            # PASSWORD ANTIGUA
            # ==================================================

            if not password_correcta:

                if password_guardada == password:

                    password_correcta = True

            # ==================================================
            # LOGIN CORRECTO
            # ==================================================

            if password_correcta:

                session.clear()

                session["usuario_id"] = (
                    resultado["id"]
                )

                session["usuario_nombre"] = (
                    resultado["nombre"]
                    or resultado["usuario"]
                    or "Usuario"
                )

                session["user"] = (
                    resultado["usuario"]
                    or resultado["nombre"]
                    or "Usuario"
                )

                logger.info(
                    f"✅ Login correcto: {usuario}"
                )

                return redirect(
                    "/inicio"
                )

        # ==================================================
        # LOGIN INCORRECTO
        # ==================================================

        return redirect(
            "/?error=invalid"
        )

    except Exception as e:

        logger.exception(
            f"❌ Error en login: {e}"
        )

        if request.is_json:

            return jsonify({

                "success": False,

                "error":
                    "Error interno del servidor"

            }), 500

        return (
            "❌ Error interno del servidor",
            500
        )


# ==========================================================
# INICIO
# ==========================================================

@app.route("/inicio")
def inicio():

    if not usuario_autenticado():

        return redirect(
            "/login"
        )

    nombre = session.get(
        "usuario_nombre",
        session.get(
            "user",
            "Usuario"
        )
    )

    ahora = datetime.now()

    return render_template(

        "inicio.html",

        nombre=nombre,

        usuario=nombre,

        fecha=ahora.strftime(
            "%Y-%m-%d"
        ),

        hora=ahora.strftime(
            "%H:%M:%S"
        )

    )


# ==========================================================
# CÁMARA
# ==========================================================

@app.route("/camara")
def camara():

    if not usuario_autenticado():

        logger.warning(
            "⚠️ Intento de acceder a cámara sin login"
        )

        return redirect(
            "/login"
        )

    return render_template(
        "index.html"
    )


# ==========================================================
# FUNCIÓN ANALIZAR IMAGEN
# ==========================================================

def analizar_imagen(frame):

    if frame is None:

        raise ValueError(
            "Imagen inválida"
        )

    logger.info(
        f"📷 Imagen recibida: {frame.shape}"
    )

    # ======================================================
    # DETECTAR ROSTRO
    # ======================================================

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

        return (
            None,
            0,
            {},
            "No se detectó ningún rostro"
        )

    logger.info(
        f"✅ Rostro detectado: {rostro.shape}"
    )

    # ======================================================
    # GUARDAR ROSTRO
    # ======================================================

    try:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        rostro_path = os.path.join(
            ROSTROS_DIR,
            f"rostro_{timestamp}.jpg"
        )

        cv2.imwrite(
            rostro_path,
            rostro
        )

        logger.info(
            f"💾 Rostro guardado: {rostro_path}"
        )

    except Exception as e:

        logger.warning(
            f"⚠️ No se pudo guardar rostro: {e}"
        )

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    logger.info(
        "🧠 PREDICIENDO EMOCIÓN..."
    )

    resultado = predict.predecir(
        rostro
    )

    # ======================================================
    # RESULTADO
    # ======================================================

    if isinstance(
        resultado,
        (tuple, list)
    ):

        if len(resultado) == 4:

            emocion = resultado[0]

            confianza = resultado[1]

            todas = resultado[2]

            consejo = resultado[3]

        elif len(resultado) == 3:

            emocion = resultado[0]

            confianza = resultado[1]

            todas = resultado[2]

            consejo = CONSEJOS.get(
                emocion,
                "Cuida de ti mismo."
            )

        else:

            raise ValueError(
                "predict.predecir() debe devolver "
                "3 o 4 valores"
            )

    else:

        raise ValueError(
            "El resultado de predict.predecir() "
            "no tiene un formato válido"
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

    return (
        emocion,
        confianza,
        todas,
        consejo
    )


# ==========================================================
# ANALIZAR EMOCIÓN
# ==========================================================

@app.route(
    "/analizar",
    methods=["POST"]
)
def analizar():

    try:

        logger.info("=" * 60)

        logger.info(
            "📥 /analizar"
        )

        logger.info(
            "🧠 INICIANDO ANÁLISIS"
        )

        # ==================================================
        # COMPROBAR SESIÓN
        # ==================================================

        usuario_id = session.get(
            "usuario_id"
        )

        usuario = session.get(
            "user"
        )

        if not usuario_id and not usuario:

            logger.warning(
                "⚠️ Análisis rechazado: usuario no autenticado"
            )

            return jsonify({

                "success": False,

                "error":
                    "Debes iniciar sesión para analizar emociones."

            }), 401

        # ==================================================
        # OBTENER JSON
        # ==================================================

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "JSON vacío"

            }), 400

        # ==================================================
        # OBTENER IMAGEN
        # ==================================================

        imagen_base64 = (
            data.get("image")
            or data.get("imagen")
        )

        if not imagen_base64:

            return jsonify({

                "success": False,

                "error":
                    "No se recibió imagen"

            }), 400

        # ==================================================
        # DECODIFICAR IMAGEN
        # ==================================================

        try:

            if "," in imagen_base64:

                imagen_base64 = (
                    imagen_base64.split(
                        ",",
                        1
                    )[1]
                )

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

            return jsonify({

                "success": False,

                "error":
                    "No se pudo decodificar la imagen"

            }), 400

        # ==================================================
        # ANALIZAR
        # ==================================================

        (
            emocion,
            confianza,
            todas,
            consejo
        ) = analizar_imagen(
            frame
        )

        # ==================================================
        # SIN ROSTRO
        # ==================================================

        if emocion is None:

            return jsonify({

                "success": False,

                "error":
                    "No se detectó ningún rostro",

                "emocion":
                    "No detectado",

                "confianza":
                    0,

                "todas":
                    {},

                "consejo":
                    consejo

            })

        # ==================================================
        # OBTENER MENSAJE
        # ==================================================

        try:

            mensaje = obtener_mensaje(
                emocion
            )

        except Exception:

            mensaje = CONSEJOS.get(
                emocion,
                "Cuida de ti mismo."
            )

        # ==================================================
        # CONVERTIR RESULTADOS A JSON
        # ==================================================

        try:

            todas_json = json.dumps(
                todas,
                ensure_ascii=False
            )

        except Exception:

            todas_json = str(
                todas
            )

        # ==================================================
        # GUARDAR ANÁLISIS
        # ==================================================

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO emociones
                (
                    usuario_id,
                    usuario,
                    emocion,
                    confianza,
                    todas_emociones,
                    mensaje
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    usuario_id,
                    usuario,
                    emocion,
                    float(confianza),
                    todas_json,
                    mensaje
                )
            )

            conn.commit()

            conn.close()

            logger.info(
                "✅ Análisis guardado en BD"
            )

        except Exception as e:

            logger.exception(
                f"❌ Error guardando análisis: {e}"
            )

        # ==================================================
        # LIBERAR MEMORIA
        # ==================================================

        gc.collect()

        logger.info(
            "✅ ANÁLISIS COMPLETADO"
        )

        logger.info("=" * 60)

        # ==================================================
        # RESPUESTA
        # ==================================================

        return jsonify({

            "success":
                True,

            "emocion":
                emocion,

            "emotion":
                emocion,

            "confianza":
                confianza,

            "confidence":
                confianza,

            "todas":
                todas,

            "advice":
                consejo,

            "consejo":
                consejo,

            "message":
                mensaje,

            "mensaje":
                mensaje

        })

    except Exception as e:

        logger.exception(
            "❌ ERROR GENERAL EN /analizar"
        )

        gc.collect()

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ==========================================================
# HISTORIAL
# ==========================================================

@app.route("/historial")
def historial():

    if not usuario_autenticado():

        return redirect(
            "/login"
        )

    try:

        conn = get_db_connection()

        usuario_id = session.get(
            "usuario_id"
        )

        usuario = session.get(
            "user"
        )

        if usuario_id:

            registros = conn.execute(
                """
                SELECT
                    id,
                    emocion,
                    confianza,
                    todas_emociones,
                    mensaje,
                    fecha
                FROM emociones
                WHERE usuario_id = ?
                   OR usuario = ?
                ORDER BY fecha DESC
                """,
                (
                    usuario_id,
                    usuario
                )
            ).fetchall()

        else:

            registros = conn.execute(
                """
                SELECT
                    id,
                    emocion,
                    confianza,
                    todas_emociones,
                    mensaje,
                    fecha
                FROM emociones
                WHERE usuario = ?
                ORDER BY fecha DESC
                """,
                (usuario,)
            ).fetchall()

        conn.close()

        nombre = session.get(
            "usuario_nombre",
            session.get(
                "user",
                "Usuario"
            )
        )

        return render_template(

            "historial.html",

            registros=registros,

            nombre=nombre

        )

    except Exception as e:

        logger.exception(
            f"❌ Error cargando historial: {e}"
        )

        return (
            "Error al cargar el historial",
            500
        )


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():

    if not usuario_autenticado():

        return redirect(
            "/login"
        )

    try:

        conn = get_db_connection()

        usuario_id = session.get(
            "usuario_id"
        )

        usuario = session.get(
            "user"
        )

        if usuario_id:

            datos = conn.execute(
                """
                SELECT
                    emocion,
                    COUNT(*) AS cantidad
                FROM emociones
                WHERE usuario_id = ?
                   OR usuario = ?
                GROUP BY emocion
                """,
                (
                    usuario_id,
                    usuario
                )
            ).fetchall()

        else:

            datos = conn.execute(
                """
                SELECT
                    emocion,
                    COUNT(*) AS cantidad
                FROM emociones
                WHERE usuario = ?
                GROUP BY emocion
                """,
                (usuario,)
            ).fetchall()

        conn.close()

        # ==================================================
        # EMOCIONES
        # ==================================================

        EMOCIONES = [
            "Enojo",
            "Felicidad",
            "Neutral",
            "Tristeza",
            "Sorpresa"
        ]

        conteo = {
            emocion: 0
            for emocion in EMOCIONES
        }

        for row in datos:

            emocion = row["emocion"]

            if emocion in conteo:

                conteo[emocion] = row["cantidad"]

        return render_template(

            "dashboard.html",

            conteo=conteo

        )

    except Exception as e:

        logger.exception(
            f"❌ Error dashboard: {e}"
        )

        return (
            "Error cargando dashboard",
            500
        )


# ==========================================================
# PÁGINA IMAGEN
# ==========================================================

@app.route(
    "/imagen",
    methods=["GET"]
)
def imagen():

    return render_template(
        "imagen.html"
    )


# ==========================================================
# PREDICCIÓN DE IMAGEN
# ==========================================================

@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    try:

        # ==================================================
        # COMPROBAR ARCHIVO
        # ==================================================

        if "imagen" not in request.files:

            return jsonify({

                "estado":
                    "error",

                "detalle":
                    "No se envió imagen"

            }), 400

        archivo = request.files[
            "imagen"
        ]

        if archivo.filename == "":

            return jsonify({

                "estado":
                    "error",

                "detalle":
                    "No se seleccionó imagen"

            }), 400

        # ==================================================
        # LEER IMAGEN
        # ==================================================

        file_bytes = np.frombuffer(
            archivo.read(),
            np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return jsonify({

                "estado":
                    "error",

                "detalle":
                    "Error al leer imagen"

            }), 400

        # ==================================================
        # ANALIZAR
        # ==================================================

        (
            emocion,
            confianza,
            todas,
            consejo
        ) = analizar_imagen(
            img
        )

        if emocion is None:

            return jsonify({

                "estado":
                    "error",

                "detalle":
                    "No se detectó ningún rostro",

                "emocion":
                    "No detectado"

            }), 400

        # ==================================================
        # MENSAJE
        # ==================================================

        try:

            mensaje = obtener_mensaje(
                emocion
            )

        except Exception:

            mensaje = CONSEJOS.get(
                emocion,
                "Cuida de ti mismo."
            )

        gc.collect()

        return jsonify({

            "estado":
                "success",

            "emocion":
                emocion,

            "confianza":
                confianza,

            "todas":
                todas,

            "consejo":
                consejo,

            "mensaje":
                mensaje

        })

    except Exception as e:

        logger.exception(
            f"❌ Error predict_image: {e}"
        )

        gc.collect()

        return jsonify({

            "estado":
                "error",

            "detalle":
                str(e)

        }), 500


# ==========================================================
# CERRAR SESIÓN
# ==========================================================

@app.route("/logout")
def logout():

    nombre = session.get(
        "usuario_nombre",
        session.get(
            "user",
            "Usuario"
        )
    )

    session.clear()

    logger.info(
        f"👋 Sesión cerrada: {nombre}"
    )

    return redirect(
        "/"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "healthy",

        "version":
            "tflite"

    }), 200


# ==========================================================
# ARCHIVOS STATIC
# ==========================================================

@app.route(
    "/static/<path:filename>"
)
def static_files(filename):

    return send_from_directory(

        os.path.join(
            BASE_DIR,
            "static"
        ),

        filename

    )


# ==========================================================
# INICIALIZAR BASE DE DATOS
# ==========================================================

init_db()


# ==========================================================
# EJECUTAR LOCALMENTE
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # CARGAR MODELO
    # ======================================================

    try:

        predict.cargar_modelo()

        logger.info(
            "✅ Modelo cargado correctamente"
        )

    except Exception as e:

        logger.exception(
            "⚠️ No se pudo cargar el modelo al iniciar"
        )

    # ======================================================
    # PUERTO
    # ======================================================

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    logger.info(
        f"🚀 Servidor iniciando en puerto {port}"
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )