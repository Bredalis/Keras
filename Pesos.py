
# Librerías

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Capa densa

capa = layers.Dense()
print("Pesos: \n", capa.weights)

# Agregacion de pesos

X = tf.ones((1, 4))
y = capa(X)

print("Pesos agregados: \n", capa.weights)