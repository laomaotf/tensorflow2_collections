import os,sys
import tensorflow as tf
import numpy as np
#sys.path.insert(0,'.')
from model.attention_layers import BahdanauAttention
class Encoder(tf.keras.Model):
    def __init__(self,network_name, network_weights):
        super(Encoder,self).__init__()

        if network_name == "vgg16":
            self.feat_extract = tf.keras.applications.VGG16(include_top=False,weights=network_weights)
        else:
            self.feat_extract = tf.keras.applications.InceptionV3(include_top = False, weights = network_weights)
        self.output_dim = self.feat_extract.layers[-1].output_shape[-1] #get output dim, channels
        self.reshape = tf.keras.layers.Reshape([-1,self.output_dim]) #reshape to (B,pixel_num, channels)
    def call(self, inputs, training=None, mask=None):
        y = self.feat_extract(inputs, training=training)
        y = self.reshape(y,training=training)
        return y




class Decoder(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 attention_units, #dim attention works with
                 encoder_dim,  #output dim of encoder
                 decoder_dim, #dim of decoder works with. hidden/cell states dim
                 embedding_dim #embedding dim of input
                 ):
        super(Decoder, self).__init__()
        #LSTM init hidden/input
        self.init_h = tf.keras.layers.Dense(
            decoder_dim,input_shape=[encoder_dim],name="encoder_res2hidden")
        self.init_c = tf.keras.layers.Dense(
            decoder_dim,input_shape=[encoder_dim],name="encoder_res2carry")
        self.lstm = tf.keras.layers.LSTMCell(
            decoder_dim,name="encoder_lstm",recurrent_initializer='glorot_uniform')

        self.dropout = tf.keras.layers.Dropout(0.5,name="encoder_dropout")
        self.pred = tf.keras.layers.Dense(
            vocab_size,input_shape=[decoder_dim],name="encoder_pred")
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,embedding_dim,name = "encoder_embedding"
        )
        self.attention = BahdanauAttention(attention_units)
        return
    def call(self, input_c, hidden, carry, encoder_output, training=False):
        #input_c : (B,1), only one char each batch
        #h: (B,hidden_units)
        #c: (B,hidden_uints)
        #encoder_output: (B,pixel_num,channels)

        #1--embedding of input_c
        input_emb = self.embedding(input_c, training=training) #(B,1,embedding_dim)
        input_emb = tf.squeeze(input_emb,axis=1)#(B,embedding_dim)
        #2--replace hidden with attention output
        context = self.attention(hidden,encoder_output,training=training)#(B,channels)
        #3--LSTM
        #concat along last dim!
        lstm_input = tf.concat((context, input_emb),axis=-1) #(B,embedding_dim + channels)
        _,(hidden_new,carry_new) = self.lstm(
            lstm_input,(hidden,carry),training=training
        )
        #4--get probs across vocabs
        output = self.dropout(hidden_new,training=training) #(B,vocab_size)
        output = self.pred(output,training=training) #(B,vocab_size)

        return output, hidden_new, carry_new

    def init_hidden_state(self,encoder_output, training):
        #initialize (h,c) with mean of encoding result
        encoder_output_mean = tf.math.reduce_mean(encoder_output,axis=1) #(B,channels)
        h = self.init_h(encoder_output_mean,training=training)
        c = self.init_c(encoder_output_mean,training=training)
        return h,c

if __name__=="__main__":
    batch_size = 3
    #max_len = 64
    vocab_size =  3096
    attn_dim = 64
    encoder_dim = 256 #hidden_dim
    decoder_dim = 128
    embedding_dim = 512
    imgs = tf.random.uniform((batch_size, 224, 224, 3))
    encoder = Encoder(backbone_pretrained="../inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
    #1--encoding input seq(images)
    encoder_output = encoder(imgs)


    decoder = Decoder(vocab_size=vocab_size,attention_units= attn_dim,
                      encoder_dim = encoder_output.shape[-1], decoder_dim=decoder_dim,
                      embedding_dim=embedding_dim)
    #2--initialize decoder: hidden states / cell states
    h,c = decoder.init_hidden_state(
        encoder_output,training=False
    )
    input_token = np.random.randint(low=0,high=vocab_size-1,size=(batch_size,1)) #one token each time
    input_token = tf.convert_to_tensor(input_token)
    preds,h,c = decoder(input_token,h,c,encoder_output)
    print(f"encoder input: {imgs.shape} output: {encoder_output.shape}")
    print(f"encoder output: (batch_size, featw x feath, model_output_channel_num")
    print(f"decoder h: {h.shape}, c: {c.shape}, output: {preds.shape}")
    print(f"decoder h/c: (batch_size, decoder_dim), output: (batch_size,vocab_size)")
