
# Libraries
import tensorflow as tf
from tensorflow.keras import layers


# Create a dense layer and show its initial weights
layer = layers.Dense(units=1)  # Specify number of units
print(f"Initial weights: {layer.weights}")


# Add data to initialize the layer weights
x = tf.ones((1, 4))
y = layer(x)


# Show weights after passing data
print(f"Weights after adding data:\n{layer.weights}")
