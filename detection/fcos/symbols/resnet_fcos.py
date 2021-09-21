
# encoding=utf8
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import re
#conv-BN-RELU
def conv_module(x, filters,kernel=3, use_bn = True, use_act = True,use_bias = False,stride=1, downsampling = False, padding='same',
                use_gn = False):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=(kernel,kernel),
                               strides=(stride,stride),padding=padding,use_bias=use_bias)(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)

    if use_gn:
        x = tfa.layers.GroupNormalization(groups=32)(x)

    if use_act:
        x = tf.keras.layers.ReLU()(x)

    if downsampling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2),padding='same')(x)
    return x

#上采样模块
def upsample_module(x, filters, kernel=3, use_bn=True, use_act=True):
    x = tf.keras.layers.Conv2DTranspose(filters=filters,kernel_size=(kernel,kernel),strides=(2,2),
                                        padding='same',use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if use_act:
        x = tf.keras.layers.ReLU()(x)
    return x

###############################
#    _init_()函数，你可以在其中执行所有与输入无关的初始化
#    build()函数，可以获得输入张量的形状，并可以进行其余的初始化
#    call()函数，构建网络结构，进行前向传播
##################################################
class ScaleExp(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaleExp,self).__init__()
        return
    def build(self, input_shape):
        self.scale = self.add_weight("scale",shape=[1],initializer=tf.keras.initializers.constant(1.0),
                                       trainable = True)
        return
    def call(self,input):
        return tf.exp(tf.multiply(self.scale,input))


def get_neck_class():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU()
    ])


def get_neck_bbox():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               strides=(1, 1), padding='same', use_bias=True),
        tfa.layers.GroupNormalization(groups=32),
        tf.keras.layers.ReLU()
    ])



def get_class_head(num_class):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=num_class, kernel_size=(1,1),
                               strides=(1, 1), padding='same', use_bias=True)
    ])

def get_centerness_head():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1),
                               strides=(1, 1), padding='same', use_bias=True)
    ])
def get_bbox_head():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=4, kernel_size=(1,1),
                               strides=(1, 1), padding='same', use_bias=True)
    ])
#最终的模型
def get_backbone_resnet50(input_shape,  freeze_backbone = False, freeze_backbone_bn=False):
    backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_shape=input_shape)
    backbone.trainable = not freeze_backbone


    if freeze_backbone_bn:
        for layer in backbone.layers:
            if re.findall("_bn",layer.name) == []:
                continue
            layer.trainable = False

    backbone_output = []
    for layer in backbone.layers:
        if layer.name in {"conv3_block4_out","conv4_block6_out","conv5_block3_out"}:
            backbone_output.append(layer.output)

    return tf.keras.Model(backbone.layers[0].input,backbone_output, name="resnet50")



def get_model(input_shape, num_class, **kwargs):
    backbone = get_backbone_resnet50(input_shape=input_shape,**kwargs)


    input = tf.keras.layers.Input(shape=input_shape)

    backbone_output = backbone(input) #C3,C4,C5
    C3,C4,C5 = backbone_output

    FPN_CHANNEL_OUT = 128
    P5 = conv_module(C5,FPN_CHANNEL_OUT,kernel=1,use_bn=False,use_act=False) #P5 is only based C5
    P5up = upsample_module(P5,1024,use_bn=False, use_act=False)
    P4 = tf.keras.layers.add([P5up, C4])
    P4 = conv_module(P4,FPN_CHANNEL_OUT,kernel=1,use_bn=False, use_act=False)
    P4up = upsample_module(P4,512,use_bn=False, use_act=False)
    P3 = tf.keras.layers.add([P4up, C3])
    P3 = conv_module(P3, FPN_CHANNEL_OUT, kernel=1,use_bn=False, use_act=False)
    P6 = conv_module(P5,FPN_CHANNEL_OUT,kernel=3,stride=2,use_bn=False, use_act=False)
    P7 = tf.keras.layers.ReLU()(P6)
    P7 = conv_module(P7,FPN_CHANNEL_OUT,kernel=3,stride=2,use_bn=False, use_act=False)

    #neck/head are both shared between each FPN scale
    class_neck_run = get_neck_class()
    bbox_neck_run = get_neck_bbox()
    class_head_run = get_class_head(num_class)
    centerness_head_run = get_centerness_head()
    bbox_head_run = get_bbox_head()


    output = {}
    branch3_class = class_neck_run(P3)
    branch3_bbox = bbox_neck_run(P3)
    output["stage_0_cls"] = class_head_run(branch3_class)
    output["stage_0_center"] = centerness_head_run(branch3_class)
    output["stage_0_reg"] = ScaleExp()(bbox_head_run(branch3_bbox))

    branch4_class = class_neck_run(P4)
    branch4_bbox = bbox_neck_run(P4)
    output["stage_1_cls"] = class_head_run(branch4_class)
    output["stage_1_center"] = centerness_head_run(branch4_class)
    output["stage_1_reg"] = ScaleExp()(bbox_head_run(branch4_bbox))


    branch5_class = class_neck_run(P5)
    branch5_bbox = bbox_neck_run(P5)
    output["stage_2_cls"] = class_head_run(branch5_class)
    output["stage_2_center"] = centerness_head_run(branch5_class)
    output["stage_2_reg"] = ScaleExp()(bbox_head_run(branch5_bbox))

    branch6_class = class_neck_run(P6)
    branch6_bbox = bbox_neck_run(P6)
    output["stage_3_cls"] = class_head_run(branch6_class)
    output["stage_3_center"] = centerness_head_run(branch6_class)
    output["stage_3_reg"] = ScaleExp()(bbox_head_run(branch6_bbox))

    branch7_class = class_neck_run(P7)
    branch7_bbox = bbox_neck_run(P7)
    output["stage_4_cls"] = class_head_run(branch7_class)
    output["stage_4_center"] = centerness_head_run(branch7_class)
    output["stage_4_reg"] = ScaleExp()(bbox_head_run(branch7_bbox))

    return tf.keras.Model(input, output, name="resnet_fcos")

def TEST_DETECTION():
    num_class = 20
    input_shape = (256,512,3)
    net = get_model(input_shape,num_class=num_class,freeze_backbone=False)
    x = np.random.uniform(-1,1,size=(4,256,512,3))
    x = tf.convert_to_tensor(x)
    outputs = net(x)
    print(net.summary())
    print("input shape: ",x.shape)
    for name in outputs.keys():
        output = outputs[name]
        print(f"output shape {name}: {output.shape}")



if __name__ == "__main__":
    TEST_DETECTION()


