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
train_data, test_data = data.split(0.05)
################################
# ['efficientnet_lite0','efficientnet_lite1','efficientnet_lite2','efficientnet_lite3','efficientnet_lite4','mobilenet_v2','resnet_50']
#################################
spec = model_spec.get("efficientnet_lite0")
spec.uri = re.sub("tfhub.dev","hub.tensorflow.google.cn",spec.uri) #GFW
model = image_classifier.create(train_data,model_spec=spec,epochs=1)

os.makedirs('output',exist_ok=True)
model.export(export_dir='output',tflite_filename='model.tflite',export_format=ExportFormat.TFLITE)
model.export(export_dir='output', export_format=ExportFormat.LABEL)
