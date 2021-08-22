import tensorflow as tf

class NMS:
    def __init__(self, pool_size=3):
        self.net = tf.keras.Sequential()
        self.net.add(
            tf.keras.layers.MaxPool2D(pool_size=pool_size,strides=1,padding='same')
        )
        return
    def __call__(self,data):
        out_data = self.net.predict(data)
        return tf.equal(data, out_data)