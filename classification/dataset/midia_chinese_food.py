import os,sys
import random
import tensorflow as tf
from .base import CLASS_DATASET_BASE

CLASS_DATASET_MIDIA_CHINESE_FOOD_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
CLASS_DATASET_MIDIA_CHINESE_FOOD_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
CLASS_DATASET_MIDIA_CHINESE_FOOD_CLASS_NUM = 208
#不做增广加载图片
def loadImage(image_path):
    data = tf.io.read_file(image_path)
    data = tf.image.decode_jpeg(data, channels=3)
    data = tf.image.resize(data, (256, 256))
    data = tf.image.central_crop(data, 0.875)
    data = (data / 255.0 - CLASS_DATASET_MIDIA_CHINESE_FOOD_MEAN) / CLASS_DATASET_MIDIA_CHINESE_FOOD_STD
    return data

class CLASS_DATASET_MIDIA_CHINESE_FOOD(CLASS_DATASET_BASE):
    mean_ = CLASS_DATASET_MIDIA_CHINESE_FOOD_MEAN
    std_ = CLASS_DATASET_MIDIA_CHINESE_FOOD_STD
    def __init__(self,
                 dataset_root,
                 split, #train, val, test
                 batch_size,
                 input_size = None  #(H,W)
                 ):
        super(CLASS_DATASET_MIDIA_CHINESE_FOOD,self).__init__("midia_chinese_food",dataset_root,
                                                              split, batch_size,input_size)
        self.input_size_ = (224,224) if input_size is None else input_size
        self.image_root_ = os.path.join(self.dataset_root_,split)
        self.paths_list_, self.classes_list_ = self.loadAnnotation(split)
        self.classes_ = list(set(self.classes_list_))
        assert len(self.classes_) == CLASS_DATASET_MIDIA_CHINESE_FOOD_CLASS_NUM, f"CLASS_DATASET_MIDIA_CHINESE_FOOD_CLASS_NUM == {CLASS_DATASET_MIDIA_CHINESE_FOOD_CLASS_NUM}"
        self.step_per_epoch_ = len(self.paths_list_)//batch_size
        return
    def loadAnnotation(self,split):
        anno_path = os.path.join(self.dataset_root_,f"{split}_list.txt")
        with open(anno_path,'r') as f:
            anno_all = f.readlines()
        paths = list(map(
            lambda x: os.path.join(self.image_root_, x.split(' ')[0].strip()), anno_all
        ))
        classes = list(map(
            lambda x: int(x.split(' ')[1].strip()), anno_all
        ))
        path2class = dict(zip(paths,classes))

        if 0:
            class2path = {}
            for path in path2class.keys():
                c = path2class[path]
                if c in class2path.keys():
                    class2path[c].append(path)
                else:
                    class2path[c] = [path]
            classes, paths = [], []
            for c in class2path.keys():
                classes.extend([c for _ in range(10)])
                paths.extend(class2path[c][0:10])
            path2class = dict(zip(paths, classes))

        ###############################################
        #shuffle globally
        paths = list(path2class.keys())
        random.shuffle(paths)
        classes = [path2class[x] for x in paths]
        return paths, classes

    def loadImage(self,image_path):
        data = tf.io.read_file(image_path)
        data = tf.image.decode_jpeg(data, channels=3)


        if self.split_ == 'train':
            data = tf.image.random_flip_left_right(data)
            data = tf.image.random_flip_up_down(data)
            data = tf.image.random_brightness(data,0.5)
            #data = tf.image.resize_with_crop_or_pad(data, self.input_size_)
            data = tf.image.resize(data, (256,256))
            data = tf.image.random_crop(data, (*self.input_size_,3))
        else:
            data = tf.image.resize(data, (256, 256))
            data = tf.image.central_crop(data,0.875)


        data = (data / 255.0 - self.mean_) / self.std_
        return data

    def getClassList(self):
        return self.classes_
    def getStepPerEpoch(self):
        return self.step_per_epoch_
    def getDataset(self):
        def _convert(image_path, image_class):
            image_data = self.loadImage(image_path)
            return image_data, image_class
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False #to allow order-altering optimizations

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.paths_list_, self.classes_list_)
        ).shuffle(1024,reshuffle_each_iteration=True)
        dataset = dataset.with_options(ignore_order)
        #dataset = dataset.repeat()
        dataset = dataset.map(
            _convert, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size_,drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset