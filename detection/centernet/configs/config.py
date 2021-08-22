from yacs.config import CfgNode as CN

_C = CN()

_C.NET = CN()
_C.NET.BACKBONE = "TOY_MODEL"
_C.NET.WEIGHTS = ""
_C.NET.FREEZE_BACKBONE = True
_C.NET.FREEZE_BACKBONE_BN = True


_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = "SGD"
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.LR = CN()
_C.TRAIN.LR.ANNEALING = "COSINE"
_C.TRAIN.LR.START = 0.1
_C.TRAIN.LR.STOP = 1e-9
_C.TRAIN.LR.MOMENTUM = 0.99

_C.TRAIN.LR.WARMUP = CN()
_C.TRAIN.LR.WARMUP.START = 1e-6
_C.TRAIN.LR.WARMUP.ITERATION = 1000

_C.TRAIN.VALIDATION_EPOCH_STEP = 5
_C.TRAIN.SAVE_EPOCH_STEP = 5

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.HM_WEIGHT = 1 #HeatMap is more important
_C.TRAIN.LOSS.WH_WEIGHT = 0.1

_C.DATA = CN()
_C.DATA.TRAIN = ""
_C.DATA.VAL = ""



def get_default_config():
    return _C.clone()
