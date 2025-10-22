import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le modèle entraîné
model = tf.keras.models.load_model("mnist_model.h5")

# Créer la fenêtre principale
root = tk.Tk()
root.title("Reconnaissance de chiffres MNIST 🧠")

canvas_width, canvas_height = 280, 280
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.grid(row=0, column=0, columnspan=2)


# Affichage de l'image avant prédiction
def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.title("Image envoyée au modèle")
    plt.axis("off")
    plt.show()

# Dessin à la souris
drawing = False
def start_draw(event):
    global drawing
    drawing = True

def stop_draw(event):
    global drawing
    drawing = False

def draw(event):
    if drawing:
        x, y = event.x, event.y
        canvas.create_oval(x, y, x+15, y+15, fill="black", outline="black")

canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

# Fonction de prédiction
def predict_digit():
    # Capture du canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1))
    # show_img(img)
    img = img.convert("L")
    # show_img(img)   
    img = ImageOps.invert(img)
    # show_img(img)
    img = img.resize((28, 28))
    # show_img(img)

    # Préparation pour prédiction
    img = np.array(img) / 255.0
    show_img(img)
    img = np.expand_dims(img, axis=-1)
    show_img(img)
    img = np.expand_dims(img, axis=0)
    # show_img(img)

    # Prédiction
    pred = model.predict(img)
    result = np.argmax(pred)
    confidence = np.max(pred)

    label_result.config(text=f"→ {result} ({confidence*100:.1f}%)")

# Boutons
btn_predict = tk.Button(root, text="🔍 Prédire", command=predict_digit, width=15)
btn_predict.grid(row=1, column=0, pady=10)

def clear_canvas():
    canvas.delete("all")
    label_result.config(text="")

btn_clear = tk.Button(root, text="🧹 Effacer", command=clear_canvas, width=15)
btn_clear.grid(row=1, column=1, pady=10)

# Label résultat
label_result = tk.Label(root, text="", font=("Arial", 24))
label_result.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
