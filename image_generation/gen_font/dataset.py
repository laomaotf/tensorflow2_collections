#encoding=utf-8
import tensorflow as tf

import json


class Dataset:
    def __init__(self, batch_size):
        super(Dataset, self).__init__()
        with open('标准字迹/data.json','r') as f:
            self.sta_data=json.load(f)
        with open('手写字体/data.json','r') as f:
            self.hw_data=json.load(f)
        with open("标准字迹/data/国标一级汉字（3755个，按拼音排序）.txt",'r',encoding='utf-8') as f:
            self.text_list=list(f.read())
    
        self.num_samples=len(self.hw_data)
        self.batch_size = batch_size


    def getDataset(self):



        def _convert(img_tgt, img_src):
            img_src /= 255
            img_tgt /= 255
            #return tf.convert_to_tensor(img_src.reshape(1,-1)), tf.convert_to_tensor(img_tgt.reshape(1,-1))
            return tf.reshape(tf.convert_to_tensor(img_src),(1,-1)), tf.reshape(tf.convert_to_tensor(img_tgt),(1,-1))
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = tf.data.Dataset.from_tensor_slices(
            ( [self.hw_data[k]['img_array'] for k in range(self.num_samples)],
              [self.sta_data[self.text_list.index(self.hw_data[k]['text'])]['img_array'] for k in range(self.num_samples)]
            )
        ).shuffle(256,reshuffle_each_iteration=True)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(
            _convert, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    def __len__(self):
        return self.num_samples


if __name__=="__main__":
    import time,cv2
    import numpy as np
    ds = Dataset(1).getDataset()
    for epoch in range(10):
        t0 = time.perf_counter()
        for bs,(src, tgt) in enumerate(ds):
            src = src.numpy().reshape((64,64))
            tgt = tgt.numpy().reshape((60,48))
            src_img = (src * 255) .astype(np.uint8)
            tgt_img = (tgt * 255).astype(np.uint8)
           #cv2.imwrite("src.jpg",src_img)
           # cv2.imwrite("tgt.jpg",tgt_img)
            cv2.imshow("src", src_img)
            cv2.imshow("tgt",tgt_img)
            cv2.waitKey(-1)
            print(bs, src.shape, src.shape)
        print(f"epoch cost {time.perf_counter() - t0} sec")


