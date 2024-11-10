
# Librerías
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Definir modelo secuencial con 3 capas
modelo = Sequential([
	layers.Dense(2, activation = "relu", name = "layer1"),
	layers.Dense(3, activation = "relu", name = "layer2"),
	layers.Dense(4, name = "layer3")
])

# Datos de entrada al modelo
X = tf.ones((3, 3))
y = modelo(X)

print(f"Resultado del modelo: {y}")

# Otro método para crear un modelo secuencial
modelo_2 = Sequential()
modelo_2.add(layers.Dense(2, activation = "relu"))
modelo_2.add(layers.Dense(3, activation = "relu"))
modelo_2.add(layers.Dense(4))

# Remover la última capa de cada modelo
modelo.pop()
modelo_2.pop()

# Mostrar resumen del modelo
print(modelo.summary())