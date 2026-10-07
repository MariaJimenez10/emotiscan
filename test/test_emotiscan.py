from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


URL = "http://127.0.0.1:10000"


def crear_driver():

    options = Options()

    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(
        options=options
    )

    return driver


# ==========================================================
# PRUEBA 1
# Verificar que EmotiScan abre correctamente
# ==========================================================

def test_abrir_emotiscan():

    driver = crear_driver()

    try:

        driver.get(URL)

        assert "EmotiScan AI" in driver.title

    finally:

        driver.quit()


# ==========================================================
# PRUEBA 2
# Verificar que existen los elementos del login
# ==========================================================

def test_elementos_login():

    driver = crear_driver()

    try:

        driver.get(URL)

        # Usuario
        usuario = driver.find_element(
            By.ID,
            "usuario"
        )

        # Contraseña
        password = driver.find_element(
            By.ID,
            "password"
        )

        # Botón
        login_btn = driver.find_element(
            By.ID,
            "login-btn"
        )

        # Verificar que existen en el DOM
        assert usuario is not None
        assert password is not None
        assert login_btn is not None

        # Verificar tipos
        assert usuario.get_attribute(
            "type"
        ) == "text"

        assert password.get_attribute(
            "type"
        ) == "password"

        assert login_btn.get_attribute(
            "type"
        ) == "submit"

    finally:

        driver.quit()


# ==========================================================
# PRUEBA 3
# Verificar que los campos aceptan información
# ==========================================================

def test_campos_login():

    driver = crear_driver()

    try:

        driver.get(URL)

        usuario = driver.find_element(
            By.ID,
            "usuario"
        )

        password = driver.find_element(
            By.ID,
            "password"
        )

        usuario.send_keys(
            "usuario_prueba"
        )

        password.send_keys(
            "123456"
        )

        assert usuario.get_attribute(
            "value"
        ) == "usuario_prueba"

        assert password.get_attribute(
            "value"
        ) == "123456"

    finally:

        driver.quit()


# ==========================================================
# PRUEBA 4
# Verificar estructura del botón de login
# ==========================================================

def test_boton_login():

    driver = crear_driver()

    try:

        driver.get(URL)

        login_btn = driver.find_element(
            By.ID,
            "login-btn"
        )

        # El botón existe
        assert login_btn is not None

        # Debe ser un botón submit
        assert login_btn.get_attribute(
            "type"
        ) == "submit"

        # Debe tener la clase del botón
        clases = login_btn.get_attribute(
            "class"
        )

        assert "btn-login" in clases

        # Debe contener el elemento btn-text
        btn_text = login_btn.find_element(
            By.ID,
            "btn-text"
        )

        assert btn_text is not None

        # Verificar el contenido HTML del botón
        contenido = login_btn.get_attribute(
            "innerHTML"
        )

        assert "Ingresar" in contenido

    finally:

        driver.quit()
