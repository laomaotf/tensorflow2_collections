# %%
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from dataset import Dataset
from net import *


# %%
# 超参
EPOCH_SIZE = 100
BATCH_SIZE = 32
LR = 0.0002


# %%

optim = keras.optimizers.Adam(learning_rate=LR)
loss_fn = keras.losses.MeanSquaredError()
dataset=Dataset(BATCH_SIZE).getDataset()
loss_metric = keras.metrics.Mean()

f,ax=plt.subplots(2,5)
show_img_list=[]
inputs  =  dataset.take(1)
for one_batch in inputs:
    one_batch = one_batch[0].numpy()
    for k in range(5):
        show_img_list.append(one_batch[k])
    break

for i,img in enumerate(show_img_list):
    ax[0][i].imshow(img[0].reshape((64,64)),cmap='gray')
show_img_s = tf.concat(show_img_list,axis=0)
plt.draw()
plt.pause(0.01)

iter = 0
for epoch_id in range(EPOCH_SIZE):
    loss_metric.reset_state()
    for batch_id, data in enumerate(dataset):
        sta_imgs, hw_imgs = data
        with tf.GradientTape() as tape:
            predicts = net(sta_imgs)
            loss=loss_fn(predicts, hw_imgs)

            trainable = net.trainable_variables
            grads = tape.gradient(loss, trainable)
            optim.apply_gradients(zip(grads,trainable))
            loss_metric.update_state(loss)
        iter += 1

        if iter % 10 == 0:
            print('epoch_id:{},iter:{},loss:{}'.format(epoch_id, iter, loss_metric.result()))
            show_predict_s=net(show_img_s)
            for i,img in enumerate(show_predict_s):
                img_s = img.numpy().reshape((60,48)) * 255
                ax[1][i].imshow(img_s.astype(np.uint8),cmap='gray')
            plt.title('epoch_id:{},iter:{},loss:{}'.format(epoch_id,iter,loss_metric.result()))
            plt.draw()
            plt.pause(0.01)
    net.save("font_gen.h5")

