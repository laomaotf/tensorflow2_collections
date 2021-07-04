import tensorflow.keras as keras

# 64,64 -> 48,60


net = keras.Sequential( [
    keras.layers.Dense(5000),
    keras.layers.Activation("tanh"),
    keras.layers.Dense(5000),
    keras.layers.Activation("tanh"),
    keras.layers.Dense(5000),
    keras.layers.Activation("tanh"),
    keras.layers.Dense(48*60),
    keras.layers.Activation("sigmoid")]
)
