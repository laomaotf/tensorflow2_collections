# -*- coding: utf-8 -*-




import json,sys
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools.logger import setup_global_logging
from PIL import Image,ImageDraw
import random
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from networks import toy_model,resnet
from datetime import datetime
from tqdm import tqdm
from tensorflow.python.ops import math_ops
from configs.config import get_default_config
import logging
from solver.lr_schedule import CLASS_COSINE_LR
from utils.nms import NMS
######################################################################################
def focal_loss(alpha=0.25, gamma=2.0, eps=1e-6, from_logist=True):
    def _focal_loss(y_true, y_pred):
        if from_logist:
            pr_pred = math_ops.sigmoid(y_pred)
        else:
            pr_pred = y_pred

        #a.只有truth中心一个点作为gt
        pt = tf.where(y_true == 1, pr_pred, 1 - pr_pred) # 预测的置信度

        log_pt = tf.math.log(pt + eps) #转换成log，交叉熵定义

        #b.相对focal loss额外增加的一个负样本权重
        neg_extra_weight = tf.pow(1-y_true,4) #对远离目标的点做额外惩罚

        weight = tf.where(y_true == 1, (1-alpha) * tf.ones_like(log_pt), alpha * tf.ones_like(log_pt) * neg_extra_weight)
        loss = -weight * tf.math.pow(1-pt, gamma)*log_pt
        #return tf.reduce_mean(loss)
        num_pos = tf.reduce_sum( tf.cast(y_true==1, tf.float32) )
        if num_pos < 1:
            #loss_sum = tf.reduce_mean(loss)
            loss_sum = tf.reduce_sum(loss) #############
        else:
            loss_sum =  tf.reduce_sum(loss) / (tf.reduce_sum( tf.cast(y_true==1, tf.float32) ))
        return loss_sum
    return _focal_loss

def l1_loss(eps = 1e-6, delta=1.0):
    #tf.losses.huber()
    def _l1_loss(y_true, y_pred, mask):
        error_abs = tf.reduce_sum(tf.abs(y_true - y_pred),axis=-1)
        mask_pos = tf.squeeze(mask)
        error_sq = tf.pow(error_abs,2)
        smooth_loss = tf.where(error_abs > delta, error_abs - 0.5, 0.5 * error_sq)
        loss = tf.where(mask_pos == 1, smooth_loss, tf.zeros_like(smooth_loss))
        return tf.reduce_sum(loss) / (tf.reduce_sum(tf.cast(mask_pos,tf.float32)) + eps)
    return _l1_loss

def centernet_loss():
    calc_fm_loss = focal_loss(from_logist=True)
    calc_wh_loss = l1_loss()
    def _loss(fm_true, fm_pred, wh_true, wh_pred, mask):
        fm_loss = calc_fm_loss(fm_true, fm_pred)
        wh_loss = calc_wh_loss(wh_true, wh_pred, mask)
        return fm_loss , wh_loss
    return _loss



def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


class TrainModel:
    def __init__(self, cfg, args):
        # 训练相关参数
        self.epochs = cfg.TRAIN.NUM_EPOCHS

        self.train_tfrec_file, self.val_tfrec_file = cfg.DATA.TRAIN, cfg.DATA.VAL

        # 获得图片宽高和字符长度基本信息
        with open(os.path.splitext(cfg.DATA.TRAIN)[0] + ".cfg.json",'r') as f:
            json_cfg = json.load(f)
        self.image_height, self.image_width, self.image_channel = json_cfg['input_size'][0],json_cfg['input_size'][1],3
        self.down_ratio = json_cfg['down_ratio']
        self.image_num = json_cfg['num_images']

        self.output_folder = args.outdir
        self.output_model_folder = os.path.join(self.output_folder, "models")
        self.output_log_folder = os.path.join(self.output_folder, "logs",datetime.now().strftime("%Y%m%d-%H%M"))
        setup_global_logging(os.path.join(self.output_log_folder,"train.log"))
        logging.info("===================START TRAIN===================================")
        logging.info(cfg)

        self.backbone = cfg.NET.BACKBONE

        self.freeze_backbone = cfg.NET.FREEZE_BACKBONE

        self.hm_loss_weight = cfg.TRAIN.LOSS.HM_WEIGHT
        self.wh_loss_weight = cfg.TRAIN.LOSS.WH_WEIGHT

        self.pretrained_weights = cfg.NET.WEIGHTS

        self.num_classes = json_cfg['num_classes']

        self.train_batch_size = self.test_batch_size = cfg.TRAIN.BATCH_SIZE

        self.validation_step = cfg.TRAIN.VALIDATION_EPOCH_STEP
        self.save_step = cfg.TRAIN.SAVE_EPOCH_STEP

        self.optimizer_name = cfg.TRAIN.OPTIMIZER
        self.learning_rate = {
            "start" : cfg.TRAIN.LR.START,
            "stop": cfg.TRAIN.LR.STOP,
            "annealing":cfg.TRAIN.LR.ANNEALING,
            "warmup_start": cfg.TRAIN.LR.WARMUP.START,
            "warmup_iteration":cfg.TRAIN.LR.WARMUP.ITERATION,
            "momentum":cfg.TRAIN.LR.MOMENTUM
        }


    def load_trainval(self):

        def _parse_function(example_proto):
            features = {
                "data": tf.io.FixedLenFeature((self.image_height, self.image_width, 3), tf.int64),
                "fm": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, self.num_classes),
                                               tf.float32),
                "wh": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, 2), tf.float32),
                "mask": tf.io.FixedLenFeature((self.image_height // self.down_ratio, self.image_width // self.down_ratio, 1), tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            parsed_features['data'] = tf.cast(parsed_features['data'], tf.uint8)
            return parsed_features


        dataset_train = tf.data.TFRecordDataset(self.train_tfrec_file)
        dataset_val = tf.data.TFRecordDataset( self.val_tfrec_file)

        dataset_train = dataset_train.map(_parse_function)
        dataset_val = dataset_val.map(_parse_function)

        return dataset_train, dataset_val


    def train(self):
        os.makedirs(self.output_model_folder, exist_ok=True)
        os.makedirs(self.output_log_folder, exist_ok=True)

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(self.image_height, self.image_width))
        print("-->类别数: {}".format(self.num_classes))


        file_writer = tf.summary.create_file_writer( os.path.join(self.output_log_folder, "params")) #子目录
        file_writer.set_as_default()

        if self.backbone.lower() == "toy_model":
            model = toy_model.get_model(input_shape=(self.image_height, self.image_width, self.image_channel),
                               num_class=self.num_classes)
        elif self.backbone.lower() == "resnet50":
            model = resnet.get_model(input_shape=(self.image_height, self.image_width, self.image_channel),
                               num_class=self.num_classes,freeze_backbone=self.freeze_backbone)
        #pip install graphviz
        #pip install pydot-ng
        tf.keras.utils.plot_model(
            model, to_file='mode_detect.png', show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=True, dpi=96
        )

        train_dataset, val_dataset = self.load_trainval()



        # 数据转换和增广
        # 输入的是tfrecord中的example，输出是二元组
        def convert_val(input_data):
            image, fm, wh,mask = input_data['data'], input_data['fm'], input_data['wh'], input_data['mask']
            #image = tf.image.resize_with_crop_or_pad(image, self.image_height,
            #                                         self.image_width)  # Add pixels of padding
            # fm = tf.image.resize_with_crop_or_pad(fm, self.image_height,
            #                                          self.image_width)  # Add pixels of padding
            #
            # wh = tf.image.resize_with_crop_or_pad(wh, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # mask = tf.image.resize_with_crop_or_pad(mask, self.image_height,
            #                                       self.image_width)  # Add pixels of padding

            image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
            image = (image - 0.5)*128
            wh = tf.math.log(wh + 1)
            return image, fm, wh,mask


        def augment(input_data):
            image, fm, wh,mask = input_data['data'], input_data['fm'], input_data['wh'], input_data['mask']

            image = tf.image.random_contrast(image,lower=0.8, upper=1.3)  # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
            image = tf.image.random_saturation(image,lower=0.8,upper=1.3)
            image = tf.image.random_hue(image, max_delta=0.5)  # Random hue


            # image = tf.image.resize_with_crop_or_pad(image, self.image_height,
            #                                          self.image_width)  # Add pixels of padding
            # fm = tf.image.resize_with_crop_or_pad(fm, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # wh = tf.image.resize_with_crop_or_pad(wh, self.image_height,
            #                                       self.image_width)  # Add pixels of padding
            #
            # mask = tf.image.resize_with_crop_or_pad(mask, self.image_height,
            #                                         self.image_width)  # Add pixels of padding

            image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image t
            image = (image - 0.5) * 128 #processing of resnet50
            wh = tf.math.log(wh + 1)
            return image, fm, wh,mask

        # dataset转换成batches
        # 这里调用函数把tfrecord转换成(data,label)二元组，作为后续fit的输入
        train_batches = (
            train_dataset
                .shuffle(self.train_batch_size * 10)
                .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.train_batch_size,drop_remainder=True)
                #.repeat() #应对2.0.0的issues(StartAbort Out of range: End of sequence),fit()中需要设置steps_per_epoch
                .prefetch(tf.data.experimental.AUTOTUNE)
        )

        val_batches = (
            val_dataset
                .map(convert_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size=self.test_batch_size,drop_remainder=True)
        )

        # loading checkpoints if existing
        start_epoch = 0
        ckpt_path = os.path.join(self.output_model_folder, "cp-{epoch:04d}.ckpt")  # 可以使用str.format
        if self.pretrained_weights != "":
            model = tf.keras.models.load_model(self.pretrained_weights) # loading model saved with save()
            logging.info('微调模型 {}, start epoch from {}'.format(self.pretrained_weights, start_epoch))
        else:
            latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
            if latest_ckpt:
                start_epoch = int(os.path.basename(latest_ckpt).split('-')[1].split('.')[0])
                model.load_weights(latest_ckpt)
                logging.info('微调模型 {}, start epoch from {}'.format(latest_ckpt, start_epoch))
            else:
                logging.info('随机初始化模型，start epoch from 0')

        calc_model_loss = centernet_loss()
        #################################
        #定义优化方案
        if self.optimizer_name.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate['start'],momentum=self.learning_rate["momentum"])
        elif self.optimizer_name.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate['start'])

        #################################
        #Leanring rate annealing
        if self.learning_rate['annealing'].lower() == "cosine":
            step_total = (self.image_num + self.train_batch_size - 1) / self.train_batch_size * self.epochs
            lrsch = CLASS_COSINE_LR(optimizer, step_total, self.learning_rate['start'], self.learning_rate['stop'],
                                    self.learning_rate['warmup_start'], self.learning_rate['warmup_iteration'])
        try:
            #todos: restore lr according to start_epoch
            iter_total = 0
            train_loss_min = 1e9
            for current_epoch in range(start_epoch,self.epochs):
                loss_in_epoch = 0.0
                loss_fm_in_epoch, loss_wh_in_epoch = [], []
                pgbar = tqdm(enumerate(train_batches))
                for iter_in_epoch,batch_train in pgbar:
                    iter_total += 1
                    lrsch.step(iter_total)
                    if iter_in_epoch > 0:
                        pgbar.set_description(f"{current_epoch+1}/{self.epochs} fm loss {np.mean(loss_fm_in_epoch):.3f} wh loss {np.mean(loss_wh_in_epoch):.3f} lr {lrsch.get_lr():.5f} ")
                    data, fm, wh, mask = batch_train
                    with tf.GradientTape() as tape:
                        #########################
                        #模型预测
                        pred_fm, pred_wh = model(data)
                        #########################
                        #计算loss
                        loss_fm, loss_wh = calc_model_loss(fm, pred_fm, wh, pred_wh, mask)
                        loss = loss_fm * self.hm_loss_weight + loss_wh * self.wh_loss_weight
                        loss_in_epoch += loss.numpy()
                        loss_fm_in_epoch += [loss_fm.numpy()]
                        loss_wh_in_epoch += [loss_wh.numpy()]
                    ############################
                    #计算梯度
                    gradients = tape.gradient(loss, model.trainable_variables)
                    #############################
                    #更新模型
                    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

                train_loss = loss_in_epoch/(self.train_batch_size * (iter_in_epoch + 1))
                train_fm_loss = np.mean(loss_fm_in_epoch)
                train_wh_loss = np.mean(loss_wh_in_epoch)
                tf.summary.scalar('TrainLoss', data=train_loss, step=current_epoch)
                tf.summary.scalar("TrainFMLoss",data=train_fm_loss,step=current_epoch)
                tf.summary.scalar("TrainWHLoss",data=train_wh_loss,step=current_epoch)
                tf.summary.scalar("LR",data=lrsch.get_lr(),step=current_epoch)
                logging.info("epoch {} train-loss {:.3e}({:.2e},{:.2e})".format(
                    current_epoch + 1, train_loss, train_fm_loss, train_wh_loss))

                if train_loss_min > train_loss:
                    logging.info(f"epooch {current_epoch+1} minimum train loss {train_loss}")
                    train_loss_min = train_loss
                    tf.keras.models.save_model(model,os.path.join(os.path.dirname(ckpt_path), 'best_model.h5'))

                if 0 == (current_epoch + 1) % self.validation_step:
                    loss_in_epoch = 0
                    loss_fm_in_epoch, loss_wh_in_epoch = 0, 0
                    for iter_in_epoch,batch_train in enumerate(val_batches):
                        data, fm, wh, mask = batch_train
                        pred_fm, pred_wh = model(data)

                        if iter_in_epoch < 3:
                            B,H,W,C = pred_fm.shape
                            one_pred = pred_fm[0].numpy()
                            m0,m1 = one_pred.min(), one_pred.max()
                            vis_pred = []
                            for c in range(C):
                                vis_pred.append(  (one_pred[:,:,c] - m0) * 255 / (m1 - m0 + 0.0001)   )
                            vis_pred = np.hstack(vis_pred).astype(np.uint8)
                            #vis_pred = Image.fromarray(vis_pred)

                            one_gt = fm[0].numpy()
                            m0, m1 = one_gt.min(), one_gt.max()
                            vis_gt = []
                            for c in range(C):
                                vis_gt.append((one_gt[:, :, c] - m0) * 255 / (m1 - m0 + 0.0001))
                            vis_gt = np.hstack(vis_gt).astype(np.uint8)
                            #vis_gt = Image.fromarray(vis_gt)
                            vis = np.vstack((vis_gt, vis_pred))
                            vis = Image.fromarray(vis)
                            vis.save(os.path.join(self.output_folder,"debug_{}.jpg".format(iter_in_epoch)))

                        loss_fm, loss_wh = calc_model_loss(fm, pred_fm, wh, pred_wh, mask)
                        loss = loss_fm + loss_wh
                        loss_in_epoch += loss
                        loss_fm_in_epoch += loss_fm
                        loss_wh_in_epoch += loss_wh
                    val_loss = loss_in_epoch/(self.test_batch_size * (iter_in_epoch + 1))
                    val_fm_loss = loss_fm_in_epoch/(self.test_batch_size * (iter_in_epoch + 1))
                    val_wh_loss = loss_wh_in_epoch / (self.test_batch_size * (iter_in_epoch + 1))
                    tf.summary.scalar('TestLoss', data=val_loss, step=current_epoch)
                    logging.info("epoch {} test-loss {:.3e}({:.2e},{:.2e})".format(
                        current_epoch + 1,val_loss, val_fm_loss, val_wh_loss))
                if (current_epoch + 1) % self.save_step == 0:
                    model.save_weights(ckpt_path.format(epoch=current_epoch))
        except KeyboardInterrupt:
            model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
            logging.info("save model before quit!")

        model.save_weights(ckpt_path.format(epoch=self.epochs + 1))
        model.save(os.path.join(os.path.dirname(ckpt_path), 'final.h5'))


    # def test_images(self):
    #     model = get_model(input_shape=(self.image_height, self.image_width, self.image_channel),
    #                        num_class=self.num_classes)
    #     get_local_max = NMS()
    #     ckpt_path = os.path.join(self.output_model_folder,  "cp-{epoch:04d}.ckpt")  # 可以使用str.format
    #     latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    #     if latest_ckpt:
    #         start_epoch = int(os.path.basename(latest_ckpt).split('-')[1].split('.')[0])
    #         model.load_weights(latest_ckpt)
    #         print('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
    #     else:
    #         print('ERR: passing resume since weights not there')
    #         return 0
    #
    #     annotations = []
    #     with open(self.annotation_file, 'r') as f:
    #         for line in f:
    #             items = line.strip().split(',')
    #             usage = int(items[0])
    #             if usage == 1:
    #                 continue  # 训练集数据
    #             image_path = os.path.join(self.image_root, items[1].strip())
    #             annotations.append((image_path, items[2:]))
    #     print("images in dataset: {}".format(len(annotations)))
    #
    #
    #     for k in range(3):
    #         path_img, objs = random.choice(annotations)
    #         img = Image.open(path_img)
    #         data = np.array(img,dtype=np.uint8)
    #
    #         data = tf.image.convert_image_dtype(data, tf.float32)
    #         data = tf.reshape(data,(1,self.image_height,self.image_width,-1))
    #
    #         hm_pred, wh_pred = model.predict(data)
    #         hm_pred = tf.sigmoid(hm_pred)
    #      #   vis = hm_pred.numpy()[0,:,:,0] * 255
    #     #    plt.imshow(vis.astype(np.uint8),cmap='gray#')
    #     #    plt.show()
    #         mask_max = get_local_max(hm_pred)
    #
    #         bboxes = []
    #         for cls in range(hm_pred.shape[-1]):
    #             for y in range(hm_pred.shape[1]):
    #                 for x in range(hm_pred.shape[2]):
    #                     prob = hm_pred[0,y,x,cls].numpy()
    #                     if prob < 0.1:
    #                         continue
    #                     if mask_max[0,y,x,cls] == False:
    #                         continue
    #                     w,h = wh_pred[0,y,x,0], wh_pred[0,y,x,1]
    #                     bboxes.append(  (x-w//2, y-h//2, x+w//2, y+h//2, prob)  )
    #
    #         if len(bboxes) > 10:
    #             bboxes = sorted(bboxes,key = lambda x: x[-1], reverse=True)[0:10]
    #
    #         draw = ImageDraw.ImageDraw(img)
    #         for (x0,y0,x1,y1,_) in bboxes:
    #             x0,x1 = int(x0 * self.down_ratio),int(x1 * self.down_ratio)
    #             y0,y1 = int(y0 * self.down_ratio), int(y1 * self.down_ratio)
    #
    #             draw.rectangle((x0,y0,x1,y1),outline="Red",width=3)
    #         plt.imshow(np.asarray(img,dtype=np.uint8))
    #         plt.show()



def main(args):
    cfg = get_default_config()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    trainer = TrainModel(cfg,args)
    trainer.train()  # 开始训练模型
    #trainer.test_images()  # 识别图片示例


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="CenterNet")
    ap.add_argument("--config_file",type=str,default="configs/resnet50.voc.yml")
    ap.add_argument("--gpu",type=str,default="0")
    ap.add_argument("--cpu",type=str,default="1")
    ap.add_argument("--outdir",type=str,default="OUTPUT")
    args = ap.parse_args()

    main(args)
