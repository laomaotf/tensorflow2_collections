import os

import tensorflow as tf

##todo: add get_config() / from_config() /build() for custom model
class CLASS_INCEPTION_V3(tf.keras.Model):
    def __init__(self,class_num,pretrained_model = None, freeze_backbone = False, **kwargs):
        super(CLASS_INCEPTION_V3,self).__init__(**kwargs)
        self.backbone_ = tf.keras.applications.InceptionV3(weights=None, include_top=False)
        if pretrained_model is not None:
            self.backbone_.load_weights(pretrained_model)
        self.backbone_.trainable = not freeze_backbone
        self.flatten_ = tf.keras.layers.GlobalMaxPooling2D()
        self.output_ = tf.keras.layers.Dense(class_num,kernel_initializer='glorot_uniform')
        return
    @tf.function
    def __call__(self, x,training):
        y = self.backbone_(x,training=training)
        y = self.flatten_(y,training=training)
        y = self.output_(y)
        return y

def GetInceptionV3(input_shape,class_num, pretrain_weight="imagenet", freeze_backbone = False):
    backbone = tf.keras.applications.InceptionV3(weights=pretrain_weight,include_top=False,input_shape=input_shape)
    backbone.trainable = not freeze_backbone
    input = tf.keras.Input(shape=input_shape)
    X = backbone(input, training = not freeze_backbone)
    X = tf.keras.layers.GlobalMaxPooling2D()(X)
    X = tf.keras.layers.Dense(class_num, kernel_initializer="glorot_uniform",name="pred_probs")(X)
    return tf.keras.Model(input,X)


if __name__ == "__main__":
    #model = GetInceptionV3((224,224,3),100,freeze_backbone=True)
    model = CLASS_INCEPTION_V3(100)
    os.makedirs("output\\food",exist_ok=True)
    model.save_weights("output\\inception.ckpt")
    model.load_weights("output\\inception.ckpt")

    print("start savedmdodel....................................")
    model.save("output\\food")