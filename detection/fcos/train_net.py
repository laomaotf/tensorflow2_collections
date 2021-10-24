import os, sys

import cv2
import tensorflow as tf
from tensorflow import keras
from symbols import resnet_fcos, fcos_loss
import json
import numpy as np
from tqdm import tqdm
import tools.tfrecord_from_voc as DATA
import logging
from tools.logger import setup_global_logging
import argparse
from configs.default import get_default_cfg

def train_net(cfg):
    with open(os.path.join(cfg.TRAIN.DATA,"dataset.json"),"r") as f:
        dataset_info = json.load(f)
    NET_INPUT_H, NET_INPUT_W = cfg.TRAIN.INPUT_SIZE
    NUM_CLASS = len(dataset_info['classes'])
    NUM_IMAGE = dataset_info['image_num']



    TRAIN_BS, TEST_BS = cfg.TRAIN.BATCH_SIZE, cfg.TEST.BATCH_SIZE
    EPOCH_TOTAL = cfg.TRAIN.EPOCH_TOTAL

    if cfg.TRAIN.OPTIMIZER.lower() == "sgd":
        LR_START = cfg.SGD.LR_START
        LR_END = cfg.SGD.LR_END
        LR_POLICY = cfg.SGD.LR_POLICY
    else:
        logging.error(f"{cfg.TRAIN.OPTIMIZER} is not supported")
        return

    os.makedirs(cfg.TRAIN.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(cfg.TEST.OUTPUT_FOLDER, exist_ok=True)

    def train_step(image_data, gts,
                   model_data, optimizer):
        with tf.GradientTape() as tape:
            preds = model_data(image_data)
            losses, loss_class, loss_reg, loss_centerness = fcos_loss.calc_loss(preds,image_data,gts)
            grads = tape.gradient(losses, model_data.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_data.trainable_variables))
        return losses, loss_class, loss_reg, loss_centerness

    def create_batches(tfrecords_dir, batch_size, flag_augment, output_shape, buffer_size=1000):
        if flag_augment:
            batches = DATA.batches_from_tfrecords(tfrecords_dir=tfrecords_dir, batch_size=batch_size,
                                                  buffer_size=buffer_size,
                                                  output_shape=output_shape, flip_prob=(0, 0.5),
                                                  rotate_prob=(0.5, -5, 5))
        else:
            batches = DATA.batches_from_tfrecords(tfrecords_dir=tfrecords_dir, batch_size=batch_size,
                                                  buffer_size=buffer_size,
                                                  output_shape=output_shape, flip_prob=None,
                                                  rotate_prob=None)

        return batches

    batches = {
        "train": create_batches(cfg.TRAIN.DATA, TRAIN_BS, True, cfg.TRAIN.INPUT_SIZE),
        "test": create_batches(cfg.TEST.DATA, TEST_BS, False, cfg.TRAIN.INPUT_SIZE),
    }

    model_data = resnet_fcos.get_model((NET_INPUT_H, NET_INPUT_W, 3), num_class=NUM_CLASS, freeze_backbone=False,
                                       freeze_backbone_bn=True)
    num_batches = (NUM_IMAGE * EPOCH_TOTAL + TRAIN_BS - 1) // TRAIN_BS

    if LR_POLICY.lower() == "linear":
        def update_weight_decay(step):
            lr = LR_START * (1 - step / num_batches)
            if lr < LR_END:
                lr = LR_END
            tf.keras.backend.set_value(optimizer.lr, lr)
            return
    else:
        logging.error(f"{cfg.SGD.LR_POLICY} is not supported")
        return
    if cfg.TRAIN.OPTIMIZER.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=LR_START, momentum=cfg.SGD.MOMENTUM)
    else:
        logging.error(f"{cfg.TRAIN.OPTIMIZER} is not supported")
        return

    batch_index = 0
    for epoch in range(EPOCH_TOTAL):
        loss_total, loss_class_total, loss_reg_total, loss_centerness_total = [], [], [], []
        pbar = tqdm(batches["train"])
        for batch_data in pbar:
            batch_index += 1
            update_weight_decay(batch_index)
            losses, loss_class, loss_reg, loss_centerness = train_step(batch_data[0], batch_data[1], model_data, optimizer)
            loss_total.append(losses.numpy())
            loss_class_total.append(loss_class.numpy())
            loss_reg_total.append(loss_reg.numpy())
            loss_centerness_total.append(loss_centerness.numpy())
            pbar.set_description(
                f"{epoch} {np.mean(loss_total)} {np.mean(loss_class_total)} {np.mean(loss_reg_total)} {np.mean(loss_centerness_total)}  {tf.keras.backend.get_value(optimizer.lr)}")
        logging.info(
            f"{epoch} {np.mean(loss_total)} {np.mean(loss_class_total)} {np.mean(loss_reg_total)} {np.mean(loss_centerness_total)}  {tf.keras.backend.get_value(optimizer.lr)}"
        )

        model_data.save(os.path.join(cfg.TRAIN.OUTPUT_FOLDER, f"EPOCH{epoch}_LOSS{np.mean(loss_total)}.h5"))

        if (epoch + 1) % cfg.TRAIN.TEST_FREQ == 0 or (epoch + 1) == cfg.TRAIN.EPOCH_TOTAL:
            num_image_to_valid = 32
            for valid_num, batch_data in enumerate(batches["test"]):
                preds = model_data(batch_data[0])
                _, SRC_H, SRC_W, _ = batch_data[0].shape
                image_all_stage = []
                for stage in range(5):
                    pred_class, pred_reg, pred_centerness = preds[f'stage_{stage}_cls'], preds[f'stage_{stage}_reg'], \
                                                            preds[f'stage_{stage}_center']
                    pred_class, pred_centerness = tf.sigmoid(pred_class), tf.sigmoid(pred_centerness)
                    pred_centerness, pred_reg = pred_centerness.numpy(), pred_reg.numpy()
                    pred_class = np.max(pred_class.numpy(), axis=-1, keepdims=True)  # using class branch
                    pred_centerness = pred_centerness * pred_class

                    B, H, W, _ = pred_centerness.shape
                    SCALE = SRC_H // H
                    topk = 10
                    if topk * 2 > H * W:
                        topk = H * W // 2
                    topv, topk = tf.nn.top_k(tf.reshape(pred_centerness, (B, -1)), k=topk)
                    mask = tf.reshape(tf.reduce_min(topv, axis=-1), (B, 1)) <= tf.reshape(pred_centerness, (B, -1))
                    indices = tf.where(mask)[:, 1]
                    xyxy = tf.gather(tf.reshape(pred_reg, (B, -1, 4)), indices, axis=1)

                    location = tf.range(0, W * H)
                    location = tf.reshape(location, (1, -1))
                    location = tf.gather(location, indices, axis=1)
                    xyxy, location = xyxy[0].numpy(), location[0].numpy()

                    data_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32).reshape(1, 1, 3)
                    image_data = (batch_data[0][0].numpy() + data_mean).astype(np.uint8)
                    for n in range(len(location)):
                        y = location[n] // W
                        x = location[n] - y * W
                        x, y = x * SCALE + SCALE // 2, y * SCALE + SCALE // 2
                        x0, y0, x1, y1 = xyxy[n]
                        x0, x1 = int(x - x0), int(x + x1)
                        y0, y1 = int(y - y0), int(y + y1)
                        x0, x1 = np.clip(x0, 0, SRC_W - 1), np.clip(x1, 0, SRC_W - 1)
                        y0, y1 = np.clip(y0, 0, SRC_H - 1), np.clip(y1, 0, SRC_H - 1)
                        cv2.rectangle(image_data, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    image_all_stage.append(image_data)
                image_all_stage = np.concatenate(image_all_stage, axis=1)
                path = os.path.join(cfg.TEST.OUTPUT_FOLDER, f"test_{valid_num}.jpg")
                cv2.imwrite(path, image_all_stage)
                if valid_num > num_image_to_valid:
                    break


if __name__ == "__main__":
    ap = argparse.ArgumentParser("fcos training")
    ap.add_argument("--config_file", type=str, default="configs/voc2007_resnet50.yml")
    args = ap.parse_args()
    cfg = get_default_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    setup_global_logging(os.path.join(cfg.TRAIN.OUTPUT_FOLDER, cfg.NAME + ".log"))
    logging.info("==============Start===============")
    logging.info(cfg)
    train_net(cfg)
    logging.info("==============Done================")
