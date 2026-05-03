from selenium import webdriver

# abrir navegador
driver = webdriver.Chrome()

# abrir la página de Flask
driver.get("http://127.0.0.1:5000")

print("Página principal cargada")

# esperar hasta que presiones ENTER
input("Presiona ENTER para cerrar el navegador...")

# cerrar navegador
driver.quit()