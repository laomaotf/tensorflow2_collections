
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    savedmodel_path = "../model/output/food/"
    tflite_path = "../model/output/food.tflite"
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)
