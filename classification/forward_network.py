import tensorflow as tf
import os,argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import time
from dataset import midia_chinese_food
from config.base import get_cfg_defaults
from model.inception_net import CLASS_INCEPTION_V3
import shutil

DEBUG_ON = False
logger = None


def get_image_loader(CFG):
    if CFG.DATASET.NAME == "midia_chinese_food":
        return midia_chinese_food.loadImage
    print(f"ERROR dataset {CFG.DATASET.NAME}")
    exit(0)

def get_class_num(CFG):
    if CFG.DATASET.NAME == "midia_chinese_food":
        return midia_chinese_food.CLASS_DATASET_MIDIA_CHINESE_FOOD_CLASS_NUM
    print(f"ERROR dataset {CFG.DATASET.NAME}")
    exit(0)

def create_model(CFG):
    model_object = CLASS_INCEPTION_V3(get_class_num(CFG),
                                      None,False)
    model_path = CFG.EVALUATION.MODEL_PATH
    if os.path.splitext(model_path)[-1] == ".ckpt":
        model_object.load_weights(model_path)
    else:
        print(f"ERROR model ext {os.path.splitext(model_path)[-1]}")
        exit(0)
    return model_object



def evaluate_model(CFG,model):
    ###################################
    #SCAN FOR IMAGES
    paths = []
    for rdir, _, names in os.walk(CFG.EVALUATION.IMAGES):
        names = list(filter(
            lambda x: os.path.splitext(x)[-1] in {'.jpg',".bmp",".png"},names
        ))
        names = list(map(
            lambda x: os.path.join(rdir,x), names
        ))
        paths.extend(names)
    ############################################
    #INFO ON DATASET
    load_image = get_image_loader(CFG)
    class_total = get_class_num(CFG)
    class_names = {}
    for c in range(class_total):
        class_names[c] = f"{c}"
    if CFG.EVALUATION.CLASS_NAMES != "" and os.path.exists(CFG.EVALUATION.CLASS_NAMES):
        df = pd.read_csv(CFG.EVALUATION.CLASS_NAMES, sep=",", header=None)
        class2names = dict(zip([int(x) for x in df.values[:, 0].tolist()], df.values[:, 1].tolist()))
        for c in class_names.keys():
            if c in class2names.keys():
                class_names[c] = class2names[c]
            else:
                print(f"WARNING: miss class name of class id {c}")
    #names = [class_names[x] for x in range(class_total)]


    classes_found = defaultdict(int)
    time_cost = []
    for path in tqdm(paths,desc="evaluation..."):
        image_data = load_image(path)
        image_data = tf.expand_dims(image_data,axis=0)
        t0 = time.time()
        preds = model(image_data,training=False)
        t1 = time.time()
        time_cost.append(t1 - t0)
        preds = tf.math.softmax(preds,axis=1).numpy()
        cid = np.argmax(preds)
        name = class_names[cid]
        output_class_dir = os.path.join(CFG.EVALUATION.OUTPUT_DIR, f"{name}")
        classes_found[name] += 1
        if classes_found[name] == 1:
            os.makedirs(output_class_dir,exist_ok=True)
        shutil.copy(path, output_class_dir)
        if CFG.EVALUATION.MAX_NUM > 0 and len(time_cost) >= CFG.EVALUATION.MAX_NUM:
            print(f"WARNING: stop before finishing all images {len(paths)}")
            break

    df = pd.DataFrame({"class":list(classes_found.keys()), "num":list(classes_found.values())})
    df = df.sort_values("num",ascending=False)
    df.to_csv(os.path.join(CFG.EVALUATION.OUTPUT_DIR,"evaluation_results.csv"),index=False)
    print(df)
    print(f"time cost: {len(time_cost) * 1.0/sum(time_cost):.1f} FPS")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="yml config file path")
    args = parser.parse_args()
    CFG = get_cfg_defaults()
    CFG.merge_from_file(args.config_file)
    CFG.freeze()
    model = create_model(CFG)
    evaluate_model(CFG,model)



