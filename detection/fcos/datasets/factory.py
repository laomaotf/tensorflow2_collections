# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from collections import defaultdict

from datasets.voc import pascal_voc

__sets = defaultdict(tuple)
# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not name.lower() in __sets.keys():
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

if __name__=="__main__":
    import cv2
    ds = get_imdb("voc_2007_trainval")
    database_all = ds.roidb()
    classename_all = ds.classes
    for k, database in enumerate(database_all):
        if k > 10:
            break
        image_path = database['image']
        image = cv2.imread(image_path,1)
        bboxes = database['boxes']
        classes = database['gt_classes']
        for (x0,y0,x1,y1),class_index in zip(bboxes,classes):
            class_name = classename_all[class_index]
            cv2.rectangle(image,(x0,y0),(x1,y1),(255,0,0),2)
            cv2.putText(image,class_name,(x0,y0),cv2.FONT_HERSHEY_PLAIN,1.0,(0,255,0))
        cv2.imshow("factory test",image)
        cv2.waitKey(1000)






