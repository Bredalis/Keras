
# Librerias

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Modelo secuencial
# de 3 capas profundas

modelo = keras.Sequential(
	[
		layers.Dense(2, activation = "relu", name = "layer1"),
		layers.Dense(3, activation = "relu", name = "layer2"),
		layers.Dense(4, name = "layer3")
	]
)

# Datos ingresados al modelo

X = tf.ones((3, 3))
y = modelo(X)

print(y)

# Otro metodo para crear un modelo

modelo_2 = keras.Sequential()
modelo_2.add(layers.Dense(2, activation = "relu"))
modelo_2.add(layers.Dense(3, activation = "relu"))
modelo_2.add(layers.Dense(4))

# Borrar la ultima capa

modelo.pop()
modelo_2.pop()

# Mostrar contenido

print(modelo.summary())