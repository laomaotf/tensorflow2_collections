# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import scipy.sparse

def bbox_overlaps(boxes,query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N,K =  boxes.shape[0], query_boxes.shape[0]
    overlaps = np.zeros((N,K))
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._classes = []
        self._image_index = []
        self._roidb = None
        self._roidb_handle = None
        # Use this dict for storing dataset specific config options
        self.config = {}
        self.config['dataset_root_dir'] = os.environ["DATASET_ROOT_DIR"]
        self._cache_dir = "../train_data/cache"

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb()
        self._roidb = self.roidb_handler()
        return self._roidb()

    @property
    def cache_path(self):
        cache_path = osp.abspath(self._cache_dir)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """Evaluate detection proposal recall metrics.
        Returns:
            results: dictionary of results with keys
                'ar': average recall
                'recalls': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = { 'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                  '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_ranges = [ [0**2, 1e5**2],    # all
                        [0**2, 32**2],     # small
                        [32**2, 96**2],    # medium
                        [96**2, 1e5**2],   # large
                        [96**2, 128**2],   # 96-128
                        [128**2, 256**2],  # 128-256
                        [256**2, 512**2],  # 256-512
                        [512**2, 1e5**2],  # 512-inf
                      ]
        assert areas.has_key(area), 'unknown area range: {}'.format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = np.zeros(0)
        num_pos = 0
        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            # (...pretty hacking :/)
            max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                # If candidate_boxes is not supplied, the default is to use the
                # non-ground-truth boxes from this roidb
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in range(gt_boxes.shape[0]):
                # find which proposal box maximally covers each gt box
                argmax_overlaps = overlaps.argmax(axis=0)
                # and get the iou amount of coverage for each gt box
                max_overlaps = overlaps.max(axis=0)
                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert(gt_ovr >= 0)
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        gt_overlaps = np.sort(gt_overlaps)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)
        recalls = np.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
                'gt_overlaps': gt_overlaps}


    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a
