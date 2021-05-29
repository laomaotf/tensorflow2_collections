from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.PRETRAINED = ()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "inception_v3"
_C.MODEL.BACKBONE.PRETRAINED = "imagenet"
_C.MODEL.CODEC = CN()
_C.MODEL.CODEC.ATTENTION_DIM = 64
_C.MODEL.CODEC.DECODER_DIM = 256
_C.MODEL.CODEC.EMBEDDING_DIM = 256

_C.DATASET = CN()
_C.DATASET.NAME = "coco"


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
_C.SOLVER.INPUT_SIZE = (0,0)


_C.SOLVER.OUTPUT_DIR = ""

_C.EVALUATION = CN()
_C.EVALUATION.CODEC = ()
_C.EVALUATION.TOKENIZER = ""
_C.EVALUATION.IMAGES = ""
_C.EVALUATION.OUTPUT_DIR = ""
_C.EVALUATION.MAX_NUM = -1

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()