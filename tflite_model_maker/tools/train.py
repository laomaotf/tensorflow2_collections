############################################################################
# pip install -q tflite-model-maker
############################################################################


import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
import re


image_path = "D:\\dataset\\midia_chinese_food\\release_data\\train\\"
data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.9)
validation_data, test_data = rest_data.split(0.5)
################################
# ['efficientnet_lite0','efficientnet_lite1','efficientnet_lite2','efficientnet_lite3','efficientnet_lite4','mobilenet_v2','resnet_50']
#################################
spec = model_spec.get("efficientnet_lite0")
spec.uri = re.sub("tfhub.dev","hub.tensorflow.google.cn",spec.uri) #GFW
model = image_classifier.create(train_data,model_spec=spec,epochs=30, train_whole_model = True,use_augmentation=True,
                                learning_rate = 0.005,shuffle=True)

os.makedirs('output',exist_ok=True)
model.export(export_dir='output',tflite_filename='efficientnet-lite0-fp32.tflite',export_format=ExportFormat.TFLITE)
model.evaluate_tflite('output/efficientnet-lite0-fp32.tflite', test_data)
