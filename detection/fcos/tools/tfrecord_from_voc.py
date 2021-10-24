import os
import sys
import math
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import logging
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from tools.image_augmentor import image_augmentor
import json

MAX_OBJECT_NUM = 64
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def example_from_image(xmlpath, imgpath, use_diff=False):
    tree_root = ET.parse(xmlpath)
    image_data = tf.io.gfile.GFile(imgpath, 'rb').read()
    tree_size = tree_root.find('size')
    width, height, depth = int(tree_size.find('width').text), int(
        tree_size.find('height').text), int(tree_size.find('depth').text)
    image_shape = np.asarray([height, width, depth], np.int32)

    objects = tree_root.findall('object')
    if not use_diff:
        objects = list(filter(lambda x: int(x.find('difficult').text) == 0, objects))

    if len(objects) < 1:
        logging.error(f"{xmlpath} is empty!")
        return None

    gts = np.zeros((len(objects), 5), np.float32)
    for k, obj in enumerate(objects):
        cid = CLASSES.index(obj.find('name').text)
        bbox = obj.find('bndbox')
        x0, y0 = float(bbox.find('xmin').text), float(bbox.find('ymin').text)
        x1, y1 = float(bbox.find('xmax').text), float(bbox.find('ymax').text)
        x0, y0 = x0 - 1, y0 - 1
        x1, y1 = x1 - 1, y1 - 1
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            logging.error(f"bbox in {xmlpath} is out-of-range {height}X{width}X{depth}")
            continue
        gts[k, :] = np.asarray([x0, y0, x1, y1, cid], np.float32)
    feature = {
        "image": _bytes_feature(image_data),
        "shape": _bytes_feature(image_shape.tobytes()),
        "gt": _bytes_feature(gts.tobytes())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_from_dataset(voc_dev_root, split, output_dir, ext=".jpg", use_diff=False, num_tfrecord=5):
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    image_root = os.path.join(voc_dev_root, "JPEGImages")
    xml_root = os.path.join(voc_dev_root, "Annotations")
    split_list = os.path.join(
        voc_dev_root, "ImageSets", "Main", split + ".txt")

    if not tf.io.gfile.exists(split_list):
        logging.error(f"{split_list} does not exist!")
        return

    images_list, xmls_list = [], []
    with open(split_list, 'r') as f:
        for line in f:
            shortname = line.strip()
            if len(shortname) < 1:
                continue
            images_list.append(os.path.join(image_root, shortname + ext))
            xmls_list.append(os.path.join(xml_root, shortname + ".xml"))
    output_paths_list = []
    if len(images_list) < num_tfrecord:
        num_tfrecord = 1
        num_example_per_tfrecord = len(images_list)
    else:
        num_example_per_tfrecord = (
                                           len(images_list) + num_tfrecord - 1) // num_tfrecord

    for index_tfrecord in range(num_tfrecord):
        output_path = f"{split}_{index_tfrecord + 1:08}-of-{num_tfrecord:08}.tfrecord"
        output_path = os.path.join(output_dir, output_path)
        output_paths_list.append(output_path)
        with tf.io.TFRecordWriter(output_path) as wrt:
            start = num_example_per_tfrecord * index_tfrecord
            end = min(start + num_example_per_tfrecord, len(images_list))
            for k in range(start, end):
                example = example_from_image(
                    xmls_list[k], images_list[k], use_diff=use_diff)
                if example is None:
                    continue
                wrt.write(example.SerializeToString())
    return output_paths_list, len(images_list)


def parse_example(data, output_shape, flip_prob=None, rotate_prob=None, pad_truth_to=100):
    feature = tf.io.parse_example(
        data,
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "shape": tf.io.FixedLenFeature([], tf.string),
            "gt": tf.io.FixedLenFeature([], tf.string)
        })
    shape = tf.io.decode_raw(feature['shape'], tf.int32)
    gt = tf.io.decode_raw(feature['gt'], tf.float32)
    shape = tf.reshape(shape, [3])
    gt = tf.reshape(gt, [-1, 5])
    image = tf.image.decode_jpeg(feature['image'], channels=3)
    image = tf.cast(tf.reshape(image, shape), tf.float32)
    image, gt = image_augmentor(image, shape, data_format="channels_last", output_shape=output_shape,
                                flip_prob=flip_prob, rotate=rotate_prob, ground_truth=gt, pad_truth_to=pad_truth_to)
    data_mean = tf.constant([103.939, 116.779, 123.68],dtype=tf.float32,shape=(1,1,3))
    image = image - data_mean
    return image, gt


#读取多个tfrecord
def batches_from_tfrecords(tfrecords_dir, batch_size, buffer_size, output_shape, **kwargs):
    filenames = []
    for tfrecord_path in os.listdir(tfrecords_dir):
        if '.tfrecord' != os.path.splitext(tfrecord_path)[-1].lower():
            continue
        tfrecord_path = os.path.join(tfrecords_dir, tfrecord_path)
        filenames.append(tfrecord_path)
    dataset = tf.data.TFRecordDataset(filenames)
    batches = dataset.map(
        lambda x: parse_example(x, output_shape, **kwargs)
    ).shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)
    return batches


def create_tfrecords(output_dir):
    dataset_root = os.environ["DATASET_ROOT_DIR"]
    voc2007_root = os.path.join(dataset_root, "VOCdevkit", "VOC2007")
    splits = ["trainval", "test"]
    splits_out = ["train", 'test']
    for split, split_out in zip(splits, splits_out):
        _, num = tfrecord_from_dataset(voc2007_root, split, os.path.join(output_dir,split_out),
                                       ext=".jpg", use_diff=False, num_tfrecord=10)
        with open(os.path.join(os.path.join(output_dir,split_out), "dataset.json"), 'w') as f:
            json.dump({"image_num": num, "classes": CLASSES, "max_objects_per_image":MAX_OBJECT_NUM}, f)
    return


def test_tfrecord(tfrecords_dir, batch_size=3):
    batches = batches_from_tfrecords(tfrecords_dir, batch_size=batch_size, buffer_size=10, output_shape=(256, 512),
                                     flip_prob=(0.0, 0.5), rotate_prob=(1.0, -5, 5), pad_truth_to=MAX_OBJECT_NUM)

    num_batch = 0
    for batch in batches:
        num_batch += 1
        n = batch[0].numpy().shape[0]
        image_data = batch[0].numpy()[n - 1]
        data_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32).reshape(1, 1, 3)
        image_data = (image_data + data_mean).astype(np.uint8)
        gts = batch[1].numpy()[n - 1].astype(np.int32)
        image_data = Image.fromarray(image_data)
        draw = ImageDraw.Draw(image_data)
        for gt in gts:
            x0,y0,x1,y1,c = gt
            if c < 0:
                continue
            draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=3)
            font = ImageFont.truetype(r"C:\Windows\Fonts\simfang.ttf", size=30)
            c = CLASSES[c]
            draw.text((x0, y0), f"{c}", fill=(0, 255, 255), font=font, stroke_width=1)
        plt.imshow(image_data)
        plt.show()
    print(f"batch: {num_batch}")


if __name__ == "__main__":
    create_tfrecords("../train_data")
    #test_tfrecord("../train_data/train")
