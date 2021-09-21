import copy

import numpy as np
import cv2,random

class FLIP_IMAGE_X:
    def __init__(self,prob = 0.5):
        self.prob = prob
        return
    def __call__(self, image, target):
        if random.random() < self.prob:
            H,W,_ = image.shape
            image = cv2.flip(image,1)
            x0 = W - target[:,2] - 1
            x1 = W - target[:,0] - 1
            target[:,0], target[:,2] = x0, x1
        return image, target


class ROTAGE_IMAGE:
    def __init__(self, prob, degree = 10):
        self.prob = prob
        self.degree = degree
        return
    def _point_affine(self,M,points):
        N,_ = points.shape
        x0,y0,x1,y1 = points[:,0], points[:,1], points[:,2],points[:,3]

        xt,yt,xb,yb = (x0 + x1)/2.0,y0,(x0+x1)/2.0,y1
        xl,yl,xr,yr = x0,(y0+y1)/2.0, x1, (y0+y1)/2.0

        points = np.stack([xt,yt,xb,yb,xl,yl,xr,yr],axis=0).transpose()
        P = np.zeros((N*4,3)) + 1
        P[:,0:2] = np.reshape(points,(-1,2))
        P = np.dot(P,M.transpose())
        P = np.round(np.reshape(P,(-1,8))).astype(np.int32)
        x0,x1 = P[:,[0,2,4,6]].min(axis=1), P[:,[0,2,4,6]].max(axis=1)
        y0,y1 = P[:,[1,3,5,7]].min(axis=1), P[:,[1,3,5,7]].max(axis=1)
        P = np.stack([x0,y0,x1,y1],axis=-1)
        return P
    def __call__(self, image, target):
        if random.random() < self.prob:
            H,W,_ = image.shape
            cx,cy = W//2, H//2
            degree = random.uniform(-self.degree, self.degree)
            M = cv2.getRotationMatrix2D((cx,cy),degree,1.0)
            image = cv2.warpAffine(image, M, (W,H), borderMode=cv2.BORDER_REPLICATE,flags=cv2.INTER_LINEAR)
            target = self._point_affine(M,target)
        return image, target


class RESIZE:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
    def _get_size(self,image):
        H,W,_ = image.shape
        scale = min( [max([W,H])  * 1.0 / self.max_size, min([W,H]) * 1.0 / self.min_size] )
        ow, oh = round(W / scale), round( H / scale )
        ow, oh = min([ow,self.max_size]), min([oh, self.max_size])
        return oh,ow
    def __call__(self, image, target):
        H,W,_ = image.shape
        nh,nw = self._get_size(image)
        new_image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0,y0,x1,y1 = np.split(target,4, axis=-1)
        x0, y0 = x0 * nw / W, y0 * nh / H
        x1, y1 = x1 * nw / W, y1 * nh / H
        new_target = np.concatenate([x0,y0,x1,y1],axis=-1)
        new_target[:,[0,2]] = np.clip(new_target[:,[0,2]],0,nw)
        new_target[:,[1,3]] = np.clip(new_target[:,[1,3]],0,nh)
        new_target = np.int32(new_target)
        return new_image, new_target

if __name__ == "__main__":
    from factory import get_imdb
    ds = get_imdb("voc_2007_trainval")
    database_all = ds.roidb()
    classename_all = ds.classes

    transform_all = [
        FLIP_IMAGE_X(0.5),
        ROTAGE_IMAGE(0.5),
        RESIZE(512,1024)
    ]

    for k, database in enumerate(database_all):
        if k > 10:
            break
        image_path = database['image']
        image_src = cv2.imread(image_path, 1)
        bboxes_src = database['boxes']
        classes = database['gt_classes']
        image = copy.deepcopy(image_src)
        bboxes = copy.deepcopy(bboxes_src)
        for (x0, y0, x1, y1), class_index in zip(bboxes, classes):
            class_name = classename_all[class_index]
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(image, class_name, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.imshow("source", image)

        image_aug = copy.deepcopy(image_src)
        bboxes_aug = copy.deepcopy(bboxes_src)

        for transform in transform_all:
            image_aug, bboxes_aug = transform(image_aug,bboxes_aug)

        for (x0, y0, x1, y1), class_index in zip(bboxes_aug, classes):
            class_name = classename_all[class_index]
            cv2.rectangle(image_aug, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.putText(image_aug, class_name, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 0))
        cv2.imshow("aug", image_aug)
        cv2.waitKey(1000)


