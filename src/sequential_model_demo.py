
# Libraries
import tensorflow as tf
from tensorflow.keras import Sequential, layers


# Define sequential model with 3 layers
model = Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)

# Input data for the model
x = tf.ones((3, 3))
y = model(x)

print(f"Model output: {y}")


# Another method to create a sequential model
model_2 = Sequential()
model_2.add(layers.Dense(2, activation="relu"))
model_2.add(layers.Dense(3, activation="relu"))
model_2.add(layers.Dense(4))

# Remove the last layer from each model
model.pop()
model_2.pop()

# Show model summary
print(model.summary())
