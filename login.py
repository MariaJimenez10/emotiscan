import tkinter as tk
from tkinter import messagebox
import json
import random

archivo = "usuarios.json"

# cargar usuarios
def cargar_usuarios():
    try:
        with open(archivo, "r") as f:
            return json.load(f)
    except:
        return {}

# guardar usuarios
def guardar_usuarios(data):
    with open(archivo, "w") as f:
        json.dump(data, f)

# registrar usuario
def registrar():
    usuario = entry_user.get()
    password = entry_pass.get()

    usuarios = cargar_usuarios()

    if usuario in usuarios:
        messagebox.showerror("Error", "El usuario ya existe")
    else:
        usuarios[usuario] = password
        guardar_usuarios(usuarios)
        messagebox.showinfo("Éxito", "Usuario registrado")

# animación de chispas
def chispas():
    canvas.delete("all")

    for i in range(40):
        x = random.randint(0,300)
        y = random.randint(0,200)

        tamaño = random.randint(2,6)

        canvas.create_oval(
            x,y,
            x+tamaño,
            y+tamaño,
            fill="yellow",
            outline=""
        )

    ventana.after(100, chispas)

# login
def login():
    usuario = entry_user.get()
    password = entry_pass.get()

    usuarios = cargar_usuarios()

    if usuario in usuarios and usuarios[usuario] == password:

        messagebox.showinfo("Login", "Bienvenido " + usuario)

        # mostrar chispas
        chispas()

        ventana.after(2000, abrir_reconocimiento)

    else:
        messagebox.showerror("Error", "Usuario o contraseña incorrectos")

def abrir_reconocimiento():
    ventana.destroy()
    iniciar_reconocimiento()

# función que abre DeepFace
def iniciar_reconocimiento():
    import reconocimiento


# ventana
ventana = tk.Tk()
ventana.title("Login Sistema IA")
ventana.geometry("300x250")

tk.Label(ventana, text="Usuario").pack()
entry_user = tk.Entry(ventana)
entry_user.pack()

tk.Label(ventana, text="Contraseña").pack()
entry_pass = tk.Entry(ventana, show="*")
entry_pass.pack()

tk.Button(ventana, text="Login", command=login).pack(pady=5)
tk.Button(ventana, text="Registrar", command=registrar).pack()

# canvas para chispas
canvas = tk.Canvas(ventana, width=300, height=120, bg="black")
canvas.pack(pady=10)

ventana.mainloop()
