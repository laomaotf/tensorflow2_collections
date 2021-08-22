# encoding=utf-8
import os,sys,re,cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import  logging
from datasets.factory import get_imdb
import json
plt.rcParams['font.sans-serif'] = ['SimHei']

INPUT_1, INPUT_2, INPUT_3, INPUT_4 = "data", "fm", "wh", "mask"

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, cls):
    radius = int(radius)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma = float(diameter) / 6.0)
    #gaussian = np.expand_dims(gaussian,axis=-1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right,cls]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    #plt.imshow(masked_gaussian,cmap='gray')
    #plt.imshow(heatmap.sum(axis=-1),cmap='gray')
    #plt.show()
    return heatmap


def create_tfrecord(output_path,dataset_name = None, input_size = None, down_ratio = 4, fm_radius = 3, do_aug=False):
    if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path),exist_ok=True)
    logging.info(f'START making {output_path} tfrecord from {dataset_name}...')
    #1-创建tfrecord
    writer = tf.io.TFRecordWriter(output_path)
    #2-解析数据，转成tfrecord
    data_imdb = get_imdb(dataset_name)
    if data_imdb is None or data_imdb.roidb is None:
        return
    if do_aug:
        data_imdb.append_flipped_images()
    input_height, input_width = input_size
    num_classes = data_imdb.num_classes - 1 #不包括背景

    with open(os.path.splitext(output_path)[0] + ".cfg.json","w") as f:
        json.dump({"num_classes":num_classes,"input_size":input_size,"down_ratio":down_ratio,"num_images":data_imdb.num_images},f)

    for index,objects in tqdm(enumerate(data_imdb.roidb)):
        path_img = data_imdb.image_path_at(index)
        img = cv2.imread(path_img,1)
        if objects['flipped']:
            img = cv2.flip(img,1)
        H,W,C = img.shape

        img = cv2.resize(img,(input_width, input_height),interpolation=cv2.INTER_AREA)

        fm = np.zeros((input_height//down_ratio,input_width//down_ratio,num_classes))
        wh = np.zeros((input_height//down_ratio,input_width//down_ratio,2))
        mask = np.zeros((input_height//down_ratio,input_width//down_ratio),dtype=np.int64)
        num_objects = len(objects['gt_classes'])
        for k in range(num_objects):
            x0,y0,x1,y1 = objects['boxes'][k]
            cls = objects['gt_classes'][k] - 1 # 不包括背景
            cx,cy = (x1 + x0 ) * 0.5 * input_width / W, (y1 + y0 ) * 0.5 * input_height / H
            ny,nx = int(cy)//down_ratio, int(cx)//down_ratio
            mask[ny,nx] = 1
            fm = draw_umich_gaussian(fm,(nx,ny),fm_radius, cls)

            w,h = (x1 - x0 + 1) * input_width / W, (y1 - y0 + 1) * input_height / H
            wh[ny,nx,0] = round(w /down_ratio)
            wh[ny, nx, 1] = round(h / down_ratio)

        # a--构造Features
        feature = dict()
        feature[INPUT_1] = _int64_feature(img.reshape([-1]))
        feature[INPUT_2] = _float_feature(fm.reshape([-1]))
        feature[INPUT_3] = _float_feature(wh.reshape([-1]))
        feature[INPUT_4] = _int64_feature(mask.reshape([-1]))
        features = tf.train.Features(feature=feature)
        # b--构造example
        example = tf.train.Example(features=features)
        # c--写入tfrecord
        writer.write(example.SerializeToString())  # write one SerializedString example each time

    #3--关闭tfrecord
    writer.close()
    logging.info(f'FINISHED making {output_path} tfrecord from {dataset_name}...')
    return


def test_tfrecord(tfrec_path):
    with open(os.path.splitext(output_path)[0] + ".cfg.json", "r") as f:
        cfg = json.load(f)

    down_ratio = cfg['down_ratio']
    height,width = cfg['input_size']
    num_classes = cfg['num_classes']
    #把读取的example解析成的字典
    @tf.function
    def _parse_function(example_proto):
        features = {
            INPUT_1: tf.io.FixedLenFeature((height,width,3), tf.int64),
            INPUT_2: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio,num_classes), tf.float32),
            INPUT_3: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 2),tf.float32),
            INPUT_4: tf.io.FixedLenFeature((height//down_ratio,width//down_ratio, 1),tf.int64),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_features[INPUT_1] = tf.cast(parsed_features[INPUT_1], tf.uint8)
        return parsed_features



    dataset = tf.data.TFRecordDataset(tfrec_path)

    # total = 0
    # for _ in dataset:
    #     total += 1
    # print('{} : {}'.format(tfrec_path, total))

    dataset = dataset.map(_parse_function)

    for num, images_features in enumerate(dataset):
        #print(num)
        #if 0 != num % 1000:
        #    continue

        image_raw = images_features[INPUT_1].numpy().squeeze()
        fm = images_features[INPUT_2].numpy().squeeze()
        wh = images_features[INPUT_3].numpy().squeeze()

        image_raw = np.uint8(image_raw)
        fm_wo_cls = np.max(fm,axis=-1)
        indices = np.argwhere(fm_wo_cls == 1.0)
        for (y,x) in indices:
            w,h = wh[y,x,:]
            x,y,w,h = int(x*down_ratio), int(y * down_ratio), int(w * down_ratio), int(h * down_ratio)
            cv2.rectangle(image_raw,(x-int(w)//2, y-int(h)//2),(x+int(w)//2,y+int(h)//2),(255,0,0),3)

        figs,axes = plt.subplots(nrows=4,ncols=(num_classes + 3)//4)
        axes = axes.flatten().tolist()
        for k in range(num_classes):
            axes[k].imshow(fm[:,:,k],cmap="gray")
            axes[k].set_title(f"fm {k+1}")
        plt.figure()
        plt.imshow(image_raw,cmap='gray')
        plt.show()

def main(tfrecord_path,**kwargs):
    if not os.path.exists(tfrecord_path):
        create_tfrecord(tfrecord_path,**kwargs)
    else:
        test_tfrecord(tfrecord_path)
    return

if __name__ == "__main__":
    logging.info("===============gen_tfrecord.py======================")
    input_size = (384,384) #固定尺寸 (H,W)
    down_ratio = 4

    dataset_name = "voc_2007_trainval"
    output_path = f"../train_data/{dataset_name}_{input_size[0]}X{input_size[1]}_{down_ratio}.tfrecord"
    main(output_path,dataset_name=dataset_name,input_size=input_size,down_ratio=down_ratio,do_aug = True)

    dataset_name = "voc_2007_test"
    output_path = f"../train_data/{dataset_name}_{input_size[0]}X{input_size[1]}_{down_ratio}.tfrecord"
    main(output_path,dataset_name=dataset_name,input_size=input_size,down_ratio=down_ratio, do_aug=False)
