import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
import numpy as np

# Charger le dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation

# Construire un petit modèle séquentiel
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# Compiler et entraîner
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Évaluer
model.evaluate(x_test, y_test)

model.save("mnist_model.h5")


###################################################################### venv\Scripts\activate  python main.py  python tkinter.py
"""
# Attention aux parametres d'affichage windows->100% pas 125%

# Prendre une image au hasard du jeu de test
i = np.random.randint(0, len(x_test))
img = x_test[i]
true_label = y_test[i]

# Prédire
pred = model.predict(img.reshape(1, 28, 28))
predicted_label = np.argmax(pred)

# Afficher
plt.imshow(img, cmap='gray')
plt.title(f"Vrai : {true_label} | Prédit : {predicted_label}")
plt.axis('off')
plt.show()
"""

