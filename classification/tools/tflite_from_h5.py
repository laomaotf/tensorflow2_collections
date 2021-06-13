
import tensorflow as tf
import os
from model.inception_net import GetInceptionV3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    ##############################################################################
    # h5 -> savedmodel -> tflite
    # 2.5.0 bug: bn/moving_mean
    # 2.3.0 bug: incompatible with expected resource
    print("tf version:",tf.__version__)
    h5_path = "output\\inception_v3_midia_chinese_food\\models\\Epoch10_Loss5.526.h5"
    tflite_path = "output/inception_v3_midia_chinese_food/models.tflite"
    savedmodel_path = os.path.dirname(tflite_path)


    #convert h5 to savedmodel
    model = GetInceptionV3((224,224,3),208,freeze_backbone=False)
    os.makedirs(os.path.dirname(tflite_path),exist_ok=True)
    model.load_weights(h5_path)
    print("start savedmdodel....................................")
    model.save(savedmodel_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    #converter._enable_tflite_resource_variables = True
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)
