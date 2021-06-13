import tensorflow as tf
import os,argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re, time, random, math, pickle
from dataset.midia_chinese_food import CLASS_DATASET_MIDIA_CHINESE_FOOD as CLASS_DATASET
from config.base import get_cfg_defaults
from model.inception_net import GetInceptionV3
from solver.lr_schedule import CLASS_COSINE_LR
from utils.stepup_logging import setup
import logging

DEBUG_ON = False



def create_dataset(CFG):
    dataset_train_object, dataset_val_object = None, None
    if CFG.DATASET.NAME == "midia_chinese_food":
        dataset_train_object = CLASS_DATASET(CFG.DATASET.ROOT,"train",CFG.SOLVER.BATCH_SIZE)
        dataset_val_object = CLASS_DATASET(CFG.DATASET.ROOT,"val",CFG.SOLVER.BATCH_SIZE)
    else:
        logging.error(f"unk dataset {CFG.DATASET.NAME}")
        exit(0)

    dataset_train = dataset_train_object.getDataset()
    dataset_val = dataset_val_object.getDataset()

    if DEBUG_ON:
        def testbed_dataset(dataset):
            imgs,classes = next(iter(dataset))
            image_data, imagee_class = imgs[0], classes[0]
            image_data = ((image_data * dataset_train_object.std_ + dataset_train_object.mean_) * 255).numpy().astype(np.uint8)
            plt.imshow(image_data)
            plt.title(f"{imagee_class}")
            plt.show()
            for epoch_num in range(10):
                t0 = time.perf_counter()
                for batch_num, (image_data, image_class) in enumerate(dataset.take(1)):
                    if epoch_num == 0 and batch_num == 0:
                        print(f"image: {image_data.shape} {image_data.dtype}")
                        print(f"class: {image_class.shape} {image_class.dtype}")
                t = round(time.perf_counter() - t0)
                print(f"epoch {epoch_num}::{t} sec")
            return
        testbed_dataset(dataset_train)
    return dataset_train_object, dataset_val_object, dataset_train,dataset_val


def create_model(CFG, dataset_train_object):
    model_object = GetInceptionV3(
        (*dataset_train_object.input_size_,3),
        len(dataset_train_object.getClassList()),CFG.MODEL.PRETRAINED,freeze_backbone=CFG.SOLVER.FREEZE_BACKBONE)
    if DEBUG_ON:
        imgs, classes = next(iter(dataset_train_object))
        preds = model_object(imgs,training=True)
        print(f"model output: {preds.shape}")
        #model_object.summary()
    return model_object



def train_model(CFG,train_dataset,
                model,
                epoch_total,
                step_per_epoch,
                val_dataset = None):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.AUTO)

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()

    optimizer = tf.keras.optimizers.Adam()

    def _train_step_one(image_data, image_class):
        with tf.GradientTape() as tape:
            preds = model(image_data,training=True)

            batch_loss = loss_object(image_class,preds)
            train_accuracy.update_state(image_class,preds)

            vars = model.trainable_variables
            grads = tape.gradient(batch_loss,vars)
            if CFG.SOLVER.CLIP_GRADIENT > 0:
                grads,_ = tf.clip_by_global_norm(grads,CFG.SOLVER.CLIP_GRADIENT)
            optimizer.apply_gradients(zip(grads,vars))
            train_loss.update_state(batch_loss)
    lr_policy = None
    if CFG.SOLVER.LR.POLICY == "cosine":
        lr_policy = CLASS_COSINE_LR(optimizer, epoch_total * step_per_epoch, CFG.SOLVER.LR.START,
                                    CFG.SOLVER.LR.FINAL, CFG.SOLVER.WARMUP.START,
                                    CFG.SOLVER.WARMUP.STEPS)
    else:
        logging.error(f"unk lr policy {CFG.SOLVER.LR.POLICY}")
        exit(0)
    for epoch_num in range(epoch_total):
        train_loss.reset_states()
        train_accuracy.reset_states()
        t0 = time.time()
        for step_num,(images, classes) in tqdm(enumerate(train_dataset),desc="training"):
            _train_step_one(images,classes)
            lr_policy.step(epoch_num * step_per_epoch + step_num)
        t1 = time.time()
        if (1+epoch_num) % 5 == 0 and not(val_dataset is None):
            validation_model(val_dataset,model)
        try:
            model.save(f"{CFG.SOLVER.OUTPUT_DIR}/models/Epoch{epoch_num + 1}_Loss{train_loss.result():.3f}.h5")
        except Exception as e:
            model.save_weights(f"{CFG.SOLVER.OUTPUT_DIR}/models/Epoch{epoch_num+1}_Loss{train_loss.result():.3f}.ckpt")
        logging.info(
            f"TRAIN epoch {epoch_num + 1}/{epoch_total}, loss {train_loss.result():.5f} accuracy {train_accuracy.result():.3f} time {(t1 - t0) / 3600.0:.3f}H"
        )
    return model

def validation_model(val_dataset,model):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.AUTO)
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    t0 = time.time()
    batch_num = 0
    for image_data, image_class in tqdm(val_dataset,desc="validation"):
        preds = model(image_data,training=False)
        batch_loss = loss_object(image_class, preds)
        batch_num += 1
        val_loss.update_state(batch_loss)
        val_accuracy.update_state(image_class,preds)
    t1 = time.time()
    logging.info(
        f"VAL Loss {val_loss.result():.5f} accuracy {val_accuracy.result():.3f} time/batch {(t1 - t0) / (3600.0*batch_num):.3f} hours"
    )
    return

def train_network(config_file):
    CFG = get_cfg_defaults()
    CFG.merge_from_file(config_file)
    CFG.freeze()
    os.makedirs(f"{CFG.SOLVER.OUTPUT_DIR}/models/",exist_ok=True)

    setup("train_network",f"{CFG.SOLVER.OUTPUT_DIR}/logs/")

    logging.info("========================train with configure===========================================")
    logging.info(f"{CFG}")
    logging.info("===================================================================")

    #################################
    #DATASET
    dataset_train_object, _, dataset_train, dataset_val = create_dataset(CFG)
    #################################
    #MODEL
    model = create_model(CFG,dataset_train_object)
    #################################
    #TRAIN
    model_object = train_model(CFG,dataset_train, model, CFG.SOLVER.EPOCH_TOTAL,
                               dataset_train_object.getStepPerEpoch(),val_dataset=dataset_val)
    #################################
    #VALIDATION
    validation_model(CFG,dataset_val, model_object)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="yml config file path")
    args = parser.parse_args()
    train_network(args.config_file)


