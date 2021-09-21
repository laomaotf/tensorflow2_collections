# encoding=utf-8
import os,sys,re,cv2
sys.path.append( os.path.join( os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import  logging
from datasets.factory import get_imdb
import json
plt.rcParams['font.sans-serif'] = ['SimHei']

INF = 999999
STRIDES = [8, 16, 32,64,128]  # stride for each scale
MINMAX_RANGE = [[-1, 64], [64, 128], [128, 256],[256,512],[512,INF]]  # size supported by each scale

INPUT_DATA, INPUT_CLASS, INPUT_REG, INPUT_CENTERNESS = "data", "class", "reg", "centerness"

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

def get_original_locations(sizes,scales,minmax_ranges = None):
    assert len(sizes) == len(scales)
    if minmax_ranges is None:
        minmax_ranges = [(-1,-1) for _ in range(len(sizes))]
    else:
        assert len(sizes) == len(minmax_ranges)
    locations, locations_range = [], []
    for (h,w), scale, minmax in zip(sizes, scales, minmax_ranges):
        xs,ys = np.arange(0,w), np.arange(0,h)
        xs,ys = np.round(np.meshgrid(xs,ys))
        xs,ys = np.reshape(xs,(-1,1)) * scale + scale/2, np.reshape(ys,(-1,1)) * scale + scale/2
        locations_one_scale = np.concatenate([xs,ys],axis=1)
        locations_range_one_scale = [minmax for _ in range(len(xs))]
        locations.extend(locations_one_scale)
        locations_range.extend(locations_range_one_scale)
    return np.asarray(locations), np.asarray(locations_range)

def gen_fcos_groundtruth(image, targets):
    H,W,_ = image.shape

    ####################################################################
    #map point under each scale to original image size
    sizes = []
    for scale in STRIDES:
        sizes.append((H//scale, W//scale))
    locations, locations_range = get_original_locations(sizes, STRIDES, MINMAX_RANGE)

    xs,ys = locations[:,0], locations[:,1]
    classes = targets[:,-1]
    bboxes = targets[:,0:-1]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes] #area of each gt
    locations2gt_area = np.reshape(areas,(1,-1)).repeat(len(locations),axis=0)

    ###############################################################
    #reg target of each location
    left = xs[:,None] - bboxes[:,0][None]
    top = ys[:,None] - bboxes[:,1][None]
    right = bboxes[:,2][None] - xs[:,None]
    bottom = bboxes[:,3][None] - ys[:,None]

    reg_target = np.stack([left,top,right,bottom],axis=-1)
    is_in_bbox = (left > 0) & (top > 0) & (right > 0) & (bottom > 0) #reg gt should be positive
    reg_target_max = reg_target.max(axis=-1)
    is_in_scale = (reg_target_max >= locations_range[:,[0]]) & (reg_target_max < locations_range[:,[1]]) # max rag should be in range

    locations2gt_area[is_in_bbox==False] = INF
    locations2gt_area[is_in_scale==False] = INF

    reg_target[is_in_scale == False] = INF
    reg_target[is_in_bbox == False] = INF


    ######################################################
    #Resolve one point corresponds to more than one gt
    locations2gt_area_min = locations2gt_area.min(axis=-1)
    locations2gt_index = locations2gt_area.argmin(axis=-1)

    #########################################################
    #generate FCOS groundtruth
    locations2class = np.reshape(classes,(1,-1)).repeat(len(locations2gt_index),axis=0)
    locations2class = locations2class[range(len(locations2gt_index)),locations2gt_index]
    locations2class[locations2gt_area_min == INF] = -1 #only class >= 0 is positive
    locations2reg = reg_target[range(len(locations)),locations2gt_index] #reg gt of each location

    locations2reg_x_min = np.min(locations2reg[:,[0,2]],axis=-1,keepdims=False)
    locations2reg_x_max = np.max(locations2reg[:,[0,2]],axis=-1,keepdims=False)
    locations2reg_y_min = np.min(locations2reg[:, [1,3]], axis=-1, keepdims=False)
    locations2reg_y_max = np.max(locations2reg[:, [1,3]], axis=-1, keepdims=False)
    locations2reg_y_min, locations2reg_x_min = np.clip(locations2reg_y_min,0.001, INF), np.clip(locations2reg_x_min,0.001,INF)
    locations2centerness = np.sqrt(  (locations2reg_x_min / locations2reg_x_max) * (locations2reg_y_min / locations2reg_y_max)  )
    locations2centerness[locations2gt_area_min == INF] = 0
    locations2centerness = locations2centerness[:,None]
    locations2class = locations2class[:,None]
    return locations2class, locations2reg,locations2centerness


def create_tfrecord(output_path,dataset_name = None, input_size = None,  do_aug=False):
    if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path),exist_ok=True)
    logging.info(f'START making {output_path} tfrecord from {dataset_name}...')
    #1-创建tfrecord
    writer = tf.io.TFRecordWriter(output_path)
    #2-解析数据，转成tfrecord
    data_imdb = get_imdb(dataset_name)
    if data_imdb is None or data_imdb.roidb is None:
        return
    # if do_aug:
    #     data_imdb.append_flipped_images()
    input_height, input_width = input_size
    num_classes = data_imdb.num_classes - 1 #不包括背景
    num_locations = 0

    for index,objects in tqdm(enumerate(data_imdb.roidb)):
        path_img = data_imdb.image_path_at(index)
        img = cv2.imread(path_img,1)
        H,W,C = img.shape

        img = cv2.resize(img,(input_width, input_height),interpolation=cv2.INTER_LINEAR)
        scale_y, scale_x = input_height * 1.0 / H, input_width * 1.0 / W
        num_objects = len(objects['gt_classes'])
        targets = []
        for k in range(num_objects):
            x0,y0,x1,y1 = objects['boxes'][k]
            cls = objects['gt_classes'][k] - 1 # 不包括背景
            targets.append((x0 * scale_x,y0 * scale_y,x1 * scale_x,y1 * scale_y,cls))
        targets = np.asarray(targets)
        classes, regs, centerness = gen_fcos_groundtruth(img, targets)
        num_locations = classes.shape[0]
        #a--构造Features
        feature = dict()
        feature[INPUT_DATA] = _int64_feature(img.reshape([-1]))
        feature[INPUT_CLASS] = _float_feature(classes.reshape([-1]))
        feature[INPUT_REG] = _float_feature(regs.reshape([-1]))
        feature[INPUT_CENTERNESS] = _float_feature(centerness.reshape([-1]))
        features = tf.train.Features(feature=feature)
        # b--构造example
        example = tf.train.Example(features=features)
        # c--写入tfrecord
        writer.write(example.SerializeToString())  # write one SerializedString example each time

    #3--关闭tfrecord
    writer.close()


    with open(os.path.splitext(output_path)[0] + ".cfg.json","w") as f:
        json.dump({"num_classes":num_classes,"input_size":input_size,"num_images":data_imdb.num_images, "num_locations":num_locations},f)

    logging.info(f'FINISHED making {output_path} tfrecord from {dataset_name}...')
    return


def test_tfrecord(tfrec_path):
    with open(os.path.splitext(output_path)[0] + ".cfg.json", "r") as f:
        cfg = json.load(f)

    height,width = cfg['input_size']
    num_classes = cfg['num_classes']
    num_locations = cfg['num_locations']
    #把读取的example解析成的字典
    @tf.function
    def _parse_function(example_proto):
        features = {
            INPUT_DATA: tf.io.FixedLenFeature((height,width,3), tf.int64),
            INPUT_CLASS: tf.io.FixedLenFeature((num_locations,1), tf.float32),
            INPUT_REG: tf.io.FixedLenFeature((num_locations, 4),tf.float32),
            INPUT_CENTERNESS: tf.io.FixedLenFeature((num_locations, 1),tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_features[INPUT_DATA] = tf.cast(parsed_features[INPUT_DATA], tf.uint8)
        return parsed_features

    dataset = tf.data.TFRecordDataset(tfrec_path)



    dataset = dataset.map(_parse_function)

    for num, images_features in enumerate(dataset):
        # print(num)
        # if 0 != num % 10:
        #     continue

        image_raw = images_features[INPUT_DATA].numpy()
        locations2class = images_features[INPUT_CLASS].numpy()
        locations2reg = images_features[INPUT_REG].numpy()
        locations2centerness = images_features[INPUT_CENTERNESS].numpy()

        image_raw = np.uint8(image_raw)
        H,W,_ = image_raw.shape

        sizes = []
        for scale in STRIDES:
            sizes.append((H // scale, W // scale))
        locations, locations_range = get_original_locations(sizes, STRIDES, None)
        indices = np.argwhere(locations2class >= 0)[:,0]
        locations = locations[indices]
        locations2class = locations2class[indices]
        locations2reg = locations2reg[indices]
        locations2centerness = locations2centerness[indices]

        heatmap = np.zeros((H,W),np.float)
        for k in range(locations.shape[0]):
            x,y = np.int(locations[k,0]), np.int(locations[k,1])
            c = np.int(locations2class[k,0])
            l,t,r,b = [np.int(x) for x in locations2reg[k,:]]
            centerness = locations2centerness[k,0]

            cv2.rectangle(image_raw,(x-l,y-t),(x+r,y+b),(255,0,0),3)
            cv2.putText(image_raw,f"C{c}", (x-l,y-t),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,0))
            heatmap[y,x] = centerness
        figs,axes = plt.subplots(nrows=1,ncols=2)
        axes = axes.flatten().tolist()
        image_raw = cv2.cvtColor(image_raw,cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_raw)
        axes[0].set_title("image")
        axes[1].imshow(heatmap,cmap='gray')
        axes[1].set_title("heatmap")
        plt.savefig(f"../train_out/images/gt_{num}.jpg")
        #plt.show()
        plt.close()

def main(tfrecord_path,**kwargs):
    if not os.path.exists(os.path.splitext(output_path)[0] + ".cfg.json"):
        create_tfrecord(tfrecord_path,**kwargs)
    else:
        test_tfrecord(tfrecord_path)
    return

if __name__ == "__main__":
    logging.info("===============gen_tfrecord.py======================")
    input_size = (384,384) #固定尺寸 (H,W)

    dataset_name = "voc_2007_trainval"
    output_path = f"../train_data/{dataset_name}_{input_size[0]}X{input_size[1]}.tfrecord"
    main(output_path,dataset_name=dataset_name,input_size=input_size,do_aug = True)

    dataset_name = "voc_2007_test"
    output_path = f"../train_data/{dataset_name}_{input_size[0]}X{input_size[1]}.tfrecord"
    main(output_path,dataset_name=dataset_name,input_size=input_size, do_aug=False)
