import os
import tensorflow as tf

def GetVGG(input_shape,class_num, pretrain_weight="imagenet", freeze_backbone = False):
    backbone = tf.keras.applications.vgg16.VGG16(weights=pretrain_weight,include_top=False,input_shape=input_shape)
    backbone.trainable = not freeze_backbone
    input = tf.keras.Input(shape=input_shape)
    X = backbone(input, training = not freeze_backbone)
    X = tf.keras.layers.GlobalMaxPooling2D()(X)
    X = tf.keras.layers.Dense(class_num, kernel_initializer="glorot_uniform",name="pred_probs")(X)
    return tf.keras.Model(input,X)

