from selenium import webdriver

driver = webdriver.Chrome()

driver.get("http://127.0.0.1:5000")

print("URL abierta:", driver.current_url)

input("Presiona ENTER para cerrar...")

driver.quit()