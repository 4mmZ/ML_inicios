#!/usr/bin/python3

import tensorflow as tf
import numpy as np


celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#las capas densas son las que tienen conexiones entre todas las neuronas entre si
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Comenzando Entrenamiento")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose= False)
print("Modelo Entrenado")


print("Una prediccion cuanto f son 100 c")
resultado = modelo.predict([100.0])
print("el resultado es" + str(resultado) + "fahrenheits")