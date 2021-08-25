import os,random,cv2
import numpy as np
import tensorflow as tf
from matplotlib import  pyplot as plt
from utils.nms import NMS
from tqdm import tqdm
model_path = "pretrained/best_model.h5"
input_dir = "demo/images/"
INPUT_WIDTH, INPUT_HEIGHT,INPUT_CHANNEL = 384,384,3
PROB_THRESH = 0.1
DOWN_RATIO = 4
CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse',
                 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor')

def demo_entry():
#    model = resnet.get_model(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL),num_class=len(CLASS_NAMES))
    get_local_max = NMS(pool_size=3)
    model = tf.keras.models.load_model(model_path)

    image_paths = []
    for rdir,_,filenames in os.walk(input_dir):
        image_names = list(filter(
            lambda x: os.path.splitext(x)[-1].lower() in {'.jpg','.jpeg','.bmp','.png'},filenames
        ))
        image_paths += list(map(lambda  x: os.path.join(rdir,x), image_names))

    pbar = tqdm(image_paths)
    for image_path in pbar:
        pbar.set_description(os.path.basename(image_path))
        img = cv2.imread(image_path,1)
        img = cv2.resize(img,(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_AREA)
        data = np.array(img ,dtype=np.uint8)
        data = tf.image.convert_image_dtype(data, tf.float32)
        data = (data - 0.5) * 128
        data = tf.reshape(data ,(1 , INPUT_HEIGHT,INPUT_WIDTH, INPUT_CHANNEL))

        hm_pred, wh_pred = model.predict(data)

        hm_pred = tf.sigmoid(hm_pred)
        mask_max = get_local_max(hm_pred).numpy().squeeze()
        hm_pred = hm_pred.numpy().squeeze()
        wh_pred = tf.exp(wh_pred).numpy().squeeze() - 1

        indices = np.argwhere(mask_max * hm_pred > PROB_THRESH)
        bboxes = []
        for (y,x,cls) in indices:
            w ,h = wh_pred[y ,x ,0], wh_pred[y ,x ,1]
            prob = hm_pred[y,x,cls]
            bboxes.append(  ( x - w / 2, y - h / 2, x + w / 2, y + h / 2, cls,prob))
        if len(bboxes) > 20:
            bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)[0:20]


        for (x0, y0, x1, y1, cls, prob) in bboxes:
            x0, x1 = int(x0 * DOWN_RATIO), int(x1 * DOWN_RATIO)
            y0, y1 = int(y0 * DOWN_RATIO), int(y1 * DOWN_RATIO)
            cv2.rectangle(img,(x0, y0),(x1, y1), (255,0,0),1)
            cv2.putText(img,f"{CLASS_NAMES[cls]}",(x0,y0), cv2.FONT_HERSHEY_PLAIN,2.0,(0,255,0),1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
if __name__ == "__main__":
    demo_entry()