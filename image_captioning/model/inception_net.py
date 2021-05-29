
import tensorflow as tf

class CLASS_INCEPTION_V3(tf.keras.Model):
    def __init__(self,class_num,pretrained_model = None, freeze_backbone = False):
        super(CLASS_INCEPTION_V3,self).__init__()
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