from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "inception_v3"
_C.MODEL.PRETRAINED = "imagenet"

_C.DATASET = CN()
_C.DATASET.NAME = "midia_chinese_food"
_C.DATASET.ROOT = "D:\\dataset\\midia_chinese_food\\release_data\\"
_C.SOLVER = CN()

_C.SOLVER.LR = CN()
_C.SOLVER.LR.POLICY = ""
_C.SOLVER.LR.START = 1e-2
_C.SOLVER.LR.FINAL = 1e-5

_C.SOLVER.WARMUP = CN()
_C.SOLVER.WARMUP.START = 1e-8
_C.SOLVER.WARMUP.STEPS = 100

_C.SOLVER.EPOCH_TOTAL = 50
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.FREEZE_BACKBONE = False
_C.SOLVER.CLIP_GRADIENT = -1

_C.SOLVER.OUTPUT_DIR = ""

_C.EVALUATION = CN()
_C.EVALUATION.MODEL_PATH = ""
_C.EVALUATION.IMAGES = ""
_C.EVALUATION.OUTPUT_DIR = ""
_C.EVALUATION.MAX_NUM = -1
_C.EVALUATION.CLASS_NAMES = ""

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()