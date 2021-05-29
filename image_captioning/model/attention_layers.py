import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    # correlation between encoding_output and hidden
    # hidden = H(hidden_raw)
    # encoding = E(encoding_raw)
    # attention = Softmax( V(Relu(hidden + encoding)) ) #(B,pixel_num,1)
    # content = reduce_sum( attention * encoding_raw, axis=1)    #(B,channels) the same as hidden_raw
    def __init__(self,units):
        super(BahdanauAttention, self).__init__()
        self.H = tf.keras.layers.Dense(
            units, name = "attention_H"
        )
        self.E = tf.keras.layers.Dense(
            units, name = "attention_E"
        )
        self.V = tf.keras.layers.Dense(
            1, name="attention_V"
        )
        return
    def call(self, hidden, encoder_output, training = None):
        #hidden: (B,channels), total information of each sequence
        #encoder_output: (B,pixel_num,channels)
        hidden_expand = tf.expand_dims(hidden,axis=1) #(B,1,dim)
        hidden_dense = self.H(hidden_expand,training=training) #(B,1,units)

        encoder_output_dense = self.E(encoder_output,training=training) #(B,pixel_num,units)

        scores = tf.nn.relu(hidden_dense + encoder_output_dense) #(B,pixel_num,uints)

        scores = self.V(scores,training=training) #(B,pixel_num,1)

        #softmax along all pixels
        attention_w = tf.nn.softmax(scores,axis=1) #(B,pixel_num,1)

        context_vector = encoder_output * attention_w #(B,pixel_num,channels)

        #attention for each channel
        context_vector = tf.reduce_sum(context_vector,axis=1) #(B,channels)
        return context_vector


if __name__=="__main__":
    ANNOTATION_UNITS = 128
    attn = BahdanauAttention(ANNOTATION_UNITS)
    BS, token_size, hidden_dim, encoder_dim = 2, 16, 512, 1024
    hidden = tf.zeros([BS,hidden_dim])
    encoding_output = tf.random.uniform((BS, token_size, encoder_dim))
    attention_output = attn(
        hidden, encoding_output
    )
    print(f"hidden shape: {hidden.shape}")
    print(f"encoding outupt shpae: {encoding_output.shape}")
    print(f">> attention output shape: {attention_output.shape}")

