# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from collections import defaultdict

from datasets.pascal_voc import pascal_voc
import numpy as np

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
    ds = get_imdb("voc_2007_trainval")
    print(ds.name)