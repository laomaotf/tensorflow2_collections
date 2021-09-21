from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = ""

_C.TRAIN = CN()
_C.TRAIN.EPOCH_TOTAL = 50
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.DATA = ""
_C.TRAIN.OPTIMIZER = "sgd"
_C.TRAIN.TEST_FREQ = 5 #epoch
_C.TRAIN.OUTPUT_FOLDER = ""

_C.SGD = CN()
_C.SGD.LR_START = 0.001
_C.SGD.LR_END  = 1e-8
_C.SGD.MOMENTUM = 0.9
_C.SGD.LR_POLICY = "linear"

_C.TEST = CN()
_C.TEST.DATA = ""
_C.TEST.BATCH_SIZE = 1
_C.TEST.OUTPUT_FOLDER = ""

def get_default_cfg():
    return _C.clone()
