
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from tqdm import tqdm
import matplotlib.pyplot as plt



#load/show images
content_path, style_path = 'images/artist.jpg', 'images/water.jpg'

INPUT_SIZE_MAX = 256

LOSS_RATIO_C2S = 1e-5
LOSS_WEIGHT_STYLE = 1e2
LOSS_WEIGHT_CONTENT= LOSS_WEIGHT_STYLE * LOSS_RATIO_C2S
LOSS_WEIGHT_VAR = 1e2

EPOCH_TOTAL = 50
STEP_EACH_EPOCH = 10
START_LR = 1.0

##################################################
#特征图的选择：底层对应细节，高层对应语义,高层特征的"印象派“效果更明显
# 内容层将提取出我们的 feature maps （特征图）
content_layers = [
    #'block1_conv1',
    #'block2_conv1',
    #'block3_conv1',
    #'block4_conv1',
    'block5_conv1'
] #

# 我们感兴趣的风格层
#每个block的第一个conv
style_layers = [
                #'block1_conv1',
                #'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
]


def load_img(path_to_img):
  max_dim = INPUT_SIZE_MAX
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)
  #方便后续操作，全部转换成的浮点数 具体参考 vgg19.preprocess_input()
  img = tf.cast(tf.image.resize(img, new_shape), tf.float32)
  img = img[tf.newaxis, :]
  return img

def imshow(image, axes, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  axes.imshow(tf.cast(image,tf.uint8))
  if title:
    axes.set_title(title)
  axes.set_axis_off()

#######################################
#Load Images
content_image = load_img(content_path)
style_image = load_img(style_path)






num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def Backbone(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False #VGG不参与训练

    #获得
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model
#
# style_extractor = vgg_layers(style_layers)
# style_outputs = style_extractor(style_image*255)
#
# #查看每层输出的统计信息
# for name, output in zip(style_layers, style_outputs):
#   print(name)
#   print("  shape: ", output.numpy().shape)
#   print("  min: ", output.numpy().min())
#   print("  max: ", output.numpy().max())
#   print("  mean: ", output.numpy().mean())
#   print()


#Gram Matrix 是描述Style的关键
#Gram Matrix： (C,H*W) * (H*W,C) = (C,C)
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

#构建一个返回风格和内容张量的模型
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.backbone = Backbone(style_layers + content_layers)
        self.backbone.trainable = False

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        "Expects float input in [0,1]"

        #1--预处理
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        #2--提取特征
        outputs = self.backbone(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        #3--style使用gram矩阵描述
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        #4-content直接用特征图
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        #以字典形式，设置输出
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

StyleContentExtractor = StyleContentModel(style_layers, content_layers)



#设置输入（待修改的图像)
image = tf.Variable(tf.cast(content_image,tf.float32))
#输入、输出图值限制在[0,1]
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
#optimizer
opt = tf.optimizers.Adam(learning_rate=START_LR)
###################################################
# loss
style_targets = StyleContentExtractor(style_image)['style']  #style gt
content_targets = StyleContentExtractor(content_image)['content'] #content gt


def style_content_loss(outputs): #L2 Loss
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    #print('style loss: ', style_loss.numpy().sum())
    style_loss *= LOSS_WEIGHT_STYLE / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    #print('content loss loss: ', content_loss.numpy().sum())
    content_loss *= LOSS_WEIGHT_CONTENT / num_content_layers
    loss = style_loss + content_loss
    return loss, content_loss, style_loss

# 通过正则化图像的高频分量来减少高频误差。 在风格转移中，这通常被称为总变分损失：
def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]
  return x_var, y_var

#高频分量对应的loss

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

#@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = StyleContentExtractor(image)
    loss, closs, sloss = style_content_loss(outputs)

  grad = tape.gradient(loss, image) #计算loss_wrt_image的梯度
  opt.apply_gradients([(grad, image)])  #使用梯度更新image
  image.assign(clip_0_1(image)) #值域限制
  return loss.numpy() , closs.numpy(), sloss.numpy()

import time
start = time.time()

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes = axes.flatten() #axes flatten
plt.ion() #动态显示的关键1
plt.show()



for n in range(EPOCH_TOTAL):
    bar = tqdm(range(STEP_EACH_EPOCH))
    for _ in bar:
        loss, closs, sloss = train_step(image)
        loss_var = LOSS_WEIGHT_VAR * total_variation_loss(image)
        bar.set_description(f"content loss {closs}, style loss {sloss} var loss {loss_var.numpy()}")
        loss += loss_var
    image_to_show = image.read_value() #读取tensor的值
    imshow(content_image,axes[0],"content GT")
    imshow(style_image,axes[1],"style GT")
    imshow(image_to_show, axes[2], "epoch {}/{}".format(n + 1,EPOCH_TOTAL))
    plt.draw() #动态显示的关键2
    plt.pause(1)
plt.savefig("result.png")
plt.ioff()


end = time.time()
print("Total time: {:.1f}".format(end-start))

