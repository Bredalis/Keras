
# Librerias

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Capa 

capa = layers.Dense()
print('Pesos: \n', capa.weights)

# Agregacion de pesos

x = tf.ones((1, 4))
y = capa(x)

print('Pesos agregados: \n', capa.weights)