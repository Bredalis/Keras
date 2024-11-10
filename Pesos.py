
# Librerías
import tensorflow as tf
from tensorflow.keras import layers

# Crear una capa densa y mostrar sus pesos iniciales
capa = layers.Dense(units = 1) # Especificar unidades para la capa densa
print(f"Pesos iniciales: {capa.weights}")

# Agregar datos para calcular los pesos de la capa
X = tf.ones((1, 4))
y = capa(X)

# Mostrar los pesos después de agregar los datos
print(f"Pesos después de agregar datos: \n{capa.weights}")