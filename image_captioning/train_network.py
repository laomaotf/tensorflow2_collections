import json

import numpy as np
import os,sys,argparse
from tqdm import tqdm
from ezlog import EZLOG
import re,time,random,math,pickle,json
import tensorflow as tf
from dataset import coco
from model.attention_layers import BahdanauAttention
import matplotlib.pyplot as plt
from config.base import get_cfg_defaults
from model import codec
from solver.lr_schedule import CLASS_COSINE_LR

DEBUG_CODES = False

logger = None


SEED = 224
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




def loadDataset(cfg):
    if cfg.DATASET.NAME == "coco":
        dataset_train_object = coco.CLASS_COCO2014(
            dataset_root=os.path.join(os.environ["DATASET_ROOT_DIR"],"coco"), split="train",
            batch_size=cfg.SOLVER.BATCH_SIZE,input_size=cfg.SOLVER.INPUT_SIZE)
        dataset_test_object = coco.CLASS_COCO2014(
            dataset_root=os.path.join(os.environ["DATASET_ROOT_DIR"],"coco"), split="val",
            batch_size=cfg.SOLVER.BATCH_SIZE,input_size=cfg.SOLVER.INPUT_SIZE,tokenizer=dataset_train_object.tokenizer)
    else:
        logger.error(f"ERROR: unknown dataset name {cfg.DATASET.NAME}")
        return None,None

    if DEBUG_CODES:
        dataset_train = dataset_train_object.getDataset()
        imgs, caps = next(iter(dataset_train))
        print(f"{imgs.shape}, {caps.shape}")
        img = imgs[0].numpy().astype(np.float32)
        img = np.clip(img * coco.CLASS_DATASET_COCO_STD + coco.CLASS_DATASET_COCO_MEAN,0,255).astype(np.uint8)
        cap = caps[0].numpy().astype(np.int32)
        cap = list(map(
            lambda x: dataset_train_object.tokenizer.index_word[x],cap))
        cap = list(filter(
            lambda x: x != '<pad>',cap
        ))
        plt.imshow(img)
        plt.title(' '.join(cap))
        plt.show()
    return dataset_train_object, dataset_test_object

def getEncoderDecoder(cfg,dataset_object):
    encoder = codec.Encoder(cfg.MODEL.BACKBONE.NAME,cfg.MODEL.BACKBONE.PRETRAINED)
    decoder = codec.Decoder(
        vocab_size=dataset_object.vocab_size_,
        attention_units=cfg.MODEL.CODEC.ATTENTION_DIM,
        encoder_dim=encoder.output_dim,
        decoder_dim=cfg.MODEL.CODEC.DECODER_DIM,
        embedding_dim=cfg.MODEL.CODEC.EMBEDDING_DIM
    )
    if len(cfg.MODEL.PRETRAINED) == 2:
        encoder_model_path, decoder_model_path = cfg.MODEL.PRETRAINED
        encoder.load_weights(encoder_model_path)
        decoder.load_weights(decoder_model_path)
        logger.info(f"Loading encoder {encoder_model_path} ")
        logger.info(f"Loading decoder {decoder_model_path} ")
    return encoder, decoder

#
#

#
def startTrain(cfg,trainset_object,encoder,decoder):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)


    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_loss = tf.keras.metrics.Mean()

    optimizer = tf.keras.optimizers.Adam()
# ################################################################################
#
# EPOCHS = 10
# STEP_PER_EPOCH = 20 #not related with dataset, lr_schedule used
# TOTAL_STEPS = EPOCHS * STEP_PER_EPOCH
# WARMUP_STEPS = TOTAL_STEPS // 2
###################################################################################
    def _train_one_step(cfg,images, captions):
        loss_one_sample = 0.0
        encoder_output = encoder(images, training=False)
        with tf.GradientTape() as tape:
            h,c = decoder.init_hidden_state(encoder_output,training=True)
            char_current = tf.expand_dims(captions[:,0],1)
            for t in range(1,trainset_object.max_len_):
                captions_target = captions[:,t]
                captions_target = tf.reshape(captions_target,[cfg.SOLVER.BATCH_SIZE])
                preds, h, c = decoder(char_current,h,c,encoder_output,training=True)

                loss_one_sample += loss_object(captions_target, preds)
                train_accuracy.update_state(captions_target,preds)
                char_current = tf.expand_dims(captions_target,1)
            #vars = encoder.trainable_variables + decoder.trainable_variables
            vars = decoder.trainable_variables #not update encoder!!
            grads = tape.gradient(loss_one_sample,vars)
            if cfg.SOLVER.CLIP_GRADIENT > 0:
                grads,_ = tf.clip_by_global_norm(grads,cfg.SOLVER.CLIP_GRADIENT)
            optimizer.apply_gradients(zip(grads,vars))

            loss_one_char = loss_one_sample / (trainset_object.max_len_ - 1)
            train_loss.update_state(loss_one_char)

    steps_per_epoch = trainset_object.getStepPerEpoch()
    lr_policy = CLASS_COSINE_LR(optimizer,
                                step_total=cfg.SOLVER.EPOCH_TOTAL * steps_per_epoch,
                                lr_start=cfg.SOLVER.LR.START,lr_final=cfg.SOLVER.LR.FINAL,
                                warmup_lr_start=cfg.SOLVER.WARMUP.START,
                                warmup_steps=cfg.SOLVER.WARMUP.STEPS)
    train_dataset = trainset_object.getDataset()
    for epoch_num in range(cfg.SOLVER.EPOCH_TOTAL):
        epoch_t0 = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        pbar = tqdm(enumerate(train_dataset))
        for batch_num, (imgs,caps) in pbar:
            _train_one_step(cfg,imgs,caps)
            lr_policy.step(epoch_num * steps_per_epoch + batch_num)
            pbar.set_description(desc=f"mean-train-loss {train_loss.result().numpy():.4f}")

        encoder.save_weights(f"{cfg.SOLVER.OUTPUT_DIR}/weights/{epoch_num + 1}ENCODER_{train_loss.result():.5f}.ckpt")
        decoder.save_weights(f"{cfg.SOLVER.OUTPUT_DIR}/weights/{epoch_num + 1}DECODER_{train_loss.result():.5f}.ckpt")
        logger.info(f"epoch {epoch_num} Loss {train_loss.result().numpy()} Time {time.time() - epoch_t0:.2f} sec")




# converts a dense to a sparse tensor
# sparse tensors are required to compute the Levenshtein distance
def dense_to_sparse(dense):
    ones = tf.ones(dense.shape)
    indices = tf.where(ones)
    values = tf.gather_nd(dense, indices)
    sparse = tf.SparseTensor(indices, values, dense.shape)

    return sparse


def startValidation(cfg,testset_object,encoder,decoder):
    # computes the levenshtein distance between the predictions and labels
    TOKEN_START = tf.constant(testset_object.tokenizer.word_index["<start>"], dtype=tf.int64)
    TOKEN_END = tf.constant(testset_object.tokenizer.word_index["<end>"], dtype=tf.int64)
    TOKEN_PAD = tf.constant(testset_object.tokenizer.word_index["<pad>"], dtype=tf.int64)

    def _levenshteinDistance(preds, gts):
        preds = tf.cast(preds, tf.int64)

        preds = tf.where(
            tf.not_equal(preds, TOKEN_START) & tf.not_equal(preds, TOKEN_END) & tf.not_equal(preds, TOKEN_PAD),
            preds, y=0)

        # gts = strategy.gather(gts, axis=0)
        gts = tf.cast(gts, tf.int64)
        lbls = tf.where(tf.not_equal(gts, TOKEN_START) & tf.not_equal(gts, TOKEN_END) & tf.not_equal(gts, TOKEN_PAD),
                        gts, y=0)

        preds_sparse = dense_to_sparse(preds)
        lbls_sparse = dense_to_sparse(lbls)

        batch_distance = tf.edit_distance(preds_sparse, lbls_sparse, normalize=False)
        mean_distance = tf.math.reduce_mean(batch_distance)

        return mean_distance

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)
    def _loss_function(real, pred):
        loss = loss_object(real,pred)
        return tf.nn.compute_average_loss(loss,global_batch_size=cfg.SOLVER.BATCH_SIZE)

    val_loss = tf.keras.metrics.Sum()
    val_loss.reset_states()

    def _validation_step(images, captions, max_len):
        total_loss = 0.0
        encoder_output = encoder(images,training=False)
        h,c = decoder.init_hidden_state(encoder_output,training=False)
        char_current = tf.expand_dims(captions[:,0],1)

        preds_seq = tf.expand_dims(captions[:,0],1)
        for t in range(1, max_len):
            preds, h,c = decoder(char_current, h, c, encoder_output, training=False)

            total_loss += _loss_function(captions[:,t], preds)

            char_current = tf.math.argmax(preds, axis=1, output_type=tf.int32)
            char_current = tf.expand_dims(char_current,axis=1)
            char_current = tf.cast(char_current,tf.int32)
            preds_seq = tf.concat([preds_seq,char_current],axis=1)
        batch_loss = total_loss / (max_len-1)
        val_loss.update_state(batch_loss)
        return preds_seq

    total_dist = []
    testset = testset_object.getDataset()
    #dataset_iter = iter(val_dataset)
    #for step in range(steps_total):
    for batch_num,(images, captions) in tqdm(enumerate(testset),desc="testing ..."):
        #images,captions = next(dataset_iter)
        preds = _validation_step(images,captions,testset_object.max_len_)
        dist = _levenshteinDistance(preds,captions)
        total_dist.append( dist )
    return np.mean(total_dist)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="image caption")
    ap.add_argument("config_path",help="configuration in yml format")
    args = ap.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    cfg.freeze()

    logger = EZLOG(os.path.splitext(os.path.basename(__file__))[0],
                   outdir=cfg.SOLVER.OUTPUT_DIR)
    os.makedirs(os.path.join(cfg.SOLVER.OUTPUT_DIR,"weights"))

    logger.info("========================train with configure===========================================")
    logger.info(f"{cfg}")
    logger.info("===================================================================")

    train_ds, test_ds = loadDataset(cfg)
    if train_ds is None or test_ds is None:
        exit(0)

    tokenizer_json = train_ds.tokenizer.to_json()
    with open(os.path.join(cfg.SOLVER.OUTPUT_DIR,"tokenizer.json"),'w') as f:
        f.write(tokenizer_json)

    encoder,decoder = getEncoderDecoder(cfg,train_ds)

    startTrain(cfg,trainset_object=train_ds,encoder=encoder,decoder=decoder)
    startValidation(cfg, testset_object=test_ds, encoder=encoder, decoder=decoder)



