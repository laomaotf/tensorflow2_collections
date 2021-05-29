import tensorflow as tf
from keras_preprocessing.text import tokenizer_from_json
import os,argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import time
from dataset import coco
from config.base import get_cfg_defaults
from model import codec
import shutil
import logging





def get_image_loader(cfg):
    if cfg.DATASET.NAME == "coco":
        return coco.loadImage
    print(f"ERROR dataset {cfg.DATASET.NAME}")
    exit(0)


def create_model(cfg,tokenizer):
    encoder = codec.Encoder(cfg.MODEL.BACKBONE.NAME, cfg.MODEL.BACKBONE.PRETRAINED)
    decoder = codec.Decoder(
        vocab_size=tokenizer.num_words,
        attention_units=cfg.MODEL.CODEC.ATTENTION_DIM,
        encoder_dim=encoder.output_dim,
        decoder_dim=cfg.MODEL.CODEC.DECODER_DIM,
        embedding_dim=cfg.MODEL.CODEC.EMBEDDING_DIM
    )
    encoder_model_path, decoder_model_path = cfg.EVALUATION.CODEC
    encoder.load_weights(encoder_model_path)
    decoder.load_weights(decoder_model_path)
    logging.info(f"Loading encoder {encoder_model_path} ")
    logging.info(f"Loading decoder {decoder_model_path} ")
    return encoder, decoder



def evaluate_model(cfg,encoder,decoder,tokenizer):
    TOKEN_START = tf.constant(tokenizer.word_index["<start>"], dtype=tf.int64)
    TOKEN_END = tf.constant(tokenizer.word_index["<end>"], dtype=tf.int64)
    TOKEN_PAD = tf.constant(tokenizer.word_index["<pad>"], dtype=tf.int64)

    ###################################
    #SCAN FOR IMAGES
    paths = []
    for rdir, _, names in os.walk(cfg.EVALUATION.IMAGES):
        names = list(filter(
            lambda x: os.path.splitext(x)[-1] in {'.jpg',".bmp",".png"},names
        ))
        names = list(map(
            lambda x: os.path.join(rdir,x), names
        ))
        paths.extend(names)
    ############################################
    #TOKENIZER
    load_image = get_image_loader(cfg)

    time_cost = []
    for path in tqdm(paths,desc="evaluation..."):
        image_data = load_image(path,cfg.SOLVER.INPUT_SIZE)
        image_data = tf.expand_dims(image_data,axis=0)
        current_char = tf.reshape(TOKEN_START,(1,1))
        preds_seq = tf.reshape(TOKEN_START,(1,1))
        t0 = time.time()
        encoder_output = encoder(image_data,training=False)
        h, c = decoder.init_hidden_state(encoder_output, training=False)
        for t in range(1, coco.MAX_LEN):
            preds, h, c = decoder(current_char, h, c, encoder_output, training=False)
            current_char = tf.math.argmax(preds, axis=1, output_type=tf.int32)
            current_char = tf.expand_dims(current_char, axis=1)
            current_char = tf.cast(current_char, tf.int64)
            preds_seq = tf.concat([preds_seq, current_char], axis=1)
            if current_char == TOKEN_END:
                break
        t1 = time.time()
        time_cost.append(t1 - t0)
        preds_text = tokenizer.sequences_to_texts(preds_seq.numpy().tolist())
        preds_text = '_'.join(preds_text[0].split(' '))
        preds_text = ''.join(list(filter(
            lambda x: x not in {'<','>'},list(preds_text)
        )))
        preds_text = '_'.join(list(filter(
            lambda x: x != 'unk', preds_text.split('_')
        )))
        preds_text = '_'.join(preds_text.split('_')[1:-1])
        path_new = os.path.join(cfg.EVALUATION.OUTPUT_DIR,preds_text+".jpg")
        shutil.copyfile(path,path_new)
        if cfg.EVALUATION.MAX_NUM > 0 and len(time_cost) >= cfg.EVALUATION.MAX_NUM:
            logging.info(f"WARNING: stop before finishing all images {len(paths)}")
            break


    logging.info(f"time cost: {len(time_cost) * 1.0/sum(time_cost):.1f} FPS")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="yml config file path")
    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    os.makedirs(cfg.EVALUATION.OUTPUT_DIR,exist_ok=True)
    with open(cfg.EVALUATION.TOKENIZER,"r") as f:
        content = f.readlines()[0]
    tokenizer = tokenizer_from_json(content)
    encoder,decoder = create_model(cfg,tokenizer)
    evaluate_model(cfg,encoder,decoder,tokenizer)



