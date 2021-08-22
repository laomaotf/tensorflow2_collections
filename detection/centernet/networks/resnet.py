# encoding=utf8
import tensorflow as tf
import numpy as np
import re
#conv-BN-RELU
def conv_module(x, filters,kernel=3, use_bn = True, use_act = True,use_bias = False,stride=1, downsampling = False, padding='same'):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel,kernel),
                               strides=(stride,stride),padding=padding,use_bias=use_bias)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if use_act:
        x = tf.keras.layers.ReLU()(x)

    if downsampling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(x)
    return x

#上采样模块
def upsample_module(x, filters, kernel=3):
    x = conv_module(x,filters,kernel=3,use_bn=True, use_act=True)
    x = tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=(kernel,kernel),strides=(2,2),
                                        padding='same',use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

#预测objectness模块
def head_objectness(x, num_class, filters=512):
    hm = conv_module(x,filters,kernel=1,stride=1,use_act=True, use_bn=False,use_bias=True)
    # 预测层不用bn/act
    hm = conv_module(hm,num_class,kernel=1,stride=1,use_act=False, use_bn=False,use_bias=True)
    return hm
#预测height-width的模块
def head_hw(x, num_class, filters=512):
    wh = conv_module(x, filters, kernel=1, stride=1,use_act=True, use_bn=False, use_bias=True)
    #预测层不用bn/act,类别无关的wh
    wh = conv_module(wh, 2, kernel=1, stride=1, use_act=False, use_bn=False, use_bias=True)
    return wh

#最终的模型
def get_model(input_shape, num_class, freeze_backbone = True, freeze_backbone_bn=True, resnet_type=""):
    input = tf.keras.layers.Input(shape=input_shape)
    backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_shape=input_shape)
#    tf.keras.applications.resnet50.preprocess_input()
    backbone.trainable = not freeze_backbone

    if freeze_backbone_bn:
        for layer in backbone.layers:
            if re.findall("_bn",layer.name) == []:
                continue
            layer.trainable = False

    x = backbone(input)
    x = upsample_module(x,128)
    x = upsample_module(x,64)
    x = upsample_module(x,32)

    #两个输入模块
    objectness_heatmap = head_objectness(x,num_class)
    hw_heatmap = head_hw(x, num_class)

    #使用tf.keras.Model()把输入和输出组合在一起
    return tf.keras.Model(input,[objectness_heatmap,hw_heatmap],name="toy_model")


def TEST_DETECTION():
    num_class = 20
    input_shape = (256,512,3)
    net = get_model(input_shape,num_class,freeze_backbone=False)
    x = np.random.uniform(-1,1,size=(4,256,512,3))
    x = tf.convert_to_tensor(x)
    objectness, hw = net(x)
    print(net.summary())
    print(x.shape, objectness.shape, hw.shape)


if __name__ == "__main__":
    TEST_DETECTION()


